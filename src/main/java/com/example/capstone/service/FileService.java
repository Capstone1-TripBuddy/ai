package com.example.capstone.service;

import com.amazonaws.HttpMethod;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.GeneratePresignedUrlRequest;
import com.amazonaws.services.s3.model.ListObjectsV2Request;
import com.amazonaws.services.s3.model.ListObjectsV2Result;
import com.amazonaws.services.s3.model.ObjectMetadata;
import com.amazonaws.services.s3.model.PutObjectRequest;
import com.amazonaws.services.s3.model.PutObjectResult;
import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectInputStream;
import com.amazonaws.services.s3.model.S3ObjectSummary;
import com.example.capstone.dto.RequestAlbumDTO;
import com.example.capstone.dto.RequestPhotoDTO;
import com.example.capstone.entity.Photo;
import com.example.capstone.entity.TravelGroup;
import com.example.capstone.entity.User;
import com.example.capstone.repository.PhotoRepository;
import com.example.capstone.repository.UserRepository;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

@Service
public class FileService {

  @Value("${cloud.aws.s3.bucket}")
  private String bucketName;

  private String rootFolder = "photos/";

  private final AmazonS3Client s3Client;
  private final PhotoRepository photoRepository;
  private final UserRepository userRepository;
  private final TravelGroupService travelGroupService;

  @Autowired
  public FileService(
          final PhotoRepository photoRepository, final AmazonS3Client s3Client,
          final UserRepository userRepository, final UserService userService,
          final TravelGroupService travelGroupService) {
    this.photoRepository = photoRepository;
    this.s3Client = s3Client;
    this.userRepository = userRepository;
    this.travelGroupService = travelGroupService;
  }


  private String generateFilePath(MultipartFile file, Long groupId, Long userId) {
    return rootFolder
            + groupId + "/"
            + userId + "/"
            + UUID.randomUUID() + file.getOriginalFilename();
  }
  private String generateFilePath(MultipartFile file, String directory) {
    return rootFolder
            + directory + "/"
            + UUID.randomUUID() + file.getOriginalFilename();
  }

  public void storeProfilePicture(User user, MultipartFile profilePicture)
          throws IOException {
    String filePath = generateFilePath(profilePicture, "profile");
    String fileUrl = storeSingleFile(profilePicture, filePath);
    System.out.println(fileUrl.length());

    userRepository.findById(user.getId()).ifPresent(u -> {
      u.setProfilePicture(fileUrl);
      userRepository.save(u);
    });
  }

  public String storeSingleFile(MultipartFile file, String filePath) throws IOException {
    // set metadata
    ObjectMetadata metadata = new ObjectMetadata();
    metadata.setContentLength(file.getSize());
    metadata.setContentType(file.getContentType());

    // store to S3
    try (InputStream inputStream = file.getInputStream()) {
      s3Client.putObject(new PutObjectRequest(bucketName, filePath, inputStream, metadata));
    } catch (IOException e) {
      throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "PhotoService: " + e);
    }

    return filePath;
  }

  public void storeFiles(RequestPhotoDTO request) throws IOException {
    Long groupId = request.getGroupId();
    Long userId = request.getUserId();

    Optional<User> user = userRepository.findById(userId);
    Optional<TravelGroup> travelGroup = travelGroupService.findGroupById(groupId);

    if (user.isEmpty()) {
      throw new IOException("PhotoService: User not found");
    }
    if (travelGroup.isEmpty()) {
      throw new IOException("PhotoService: Travel Group not found");
    }

    // List<Photo> response = new ArrayList<>();
    for (MultipartFile file : request.getPhotos()) {
      // generate file name
      String filePath = generateFilePath(file, groupId, userId);

      // store to S3
      storeSingleFile(file, filePath);

      // save to DB
      Photo photo = new Photo(
              travelGroup.get(),
              user.get(),
              file.getOriginalFilename(),
              filePath,
              file.getSize(),
              file.getContentType()
      );
      photoRepository.save(photo);
      // response.add(photo);
    }

  }

  public String generateSignedUrl(String filePath) {
    try {
      // Signed URL 요청 생성
      int duration = 60 * 60;
      GeneratePresignedUrlRequest generatePresignedUrlRequest =
              new GeneratePresignedUrlRequest(bucketName, filePath)
                      .withMethod(HttpMethod.GET) // HTTP GET 요청용 Signed URL
                      .withExpiration(new Date(System.currentTimeMillis() + duration * 1000));

      // Signed URL 생성
      URL signedUrl = s3Client.generatePresignedUrl(generatePresignedUrlRequest);
      return signedUrl.toString();

    } catch (Exception e) {
      throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR,
              "Failed to generate signed URL: " + e.getMessage(), e);
    }
  }

  public Optional<ByteArrayResource> loadFile(String filePath) throws IOException {
    S3Object s3Object = s3Client.getObject(bucketName, filePath);
    InputStream inputStream = s3Object.getObjectContent();

    // InputStream을 ByteArrayResource로 변환
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    byte[] buffer = new byte[1024];
    int len;
    while ((len = inputStream.read(buffer)) > -1 ) {
      baos.write(buffer, 0, len);
    }
    baos.flush();
    ByteArrayResource resource = new ByteArrayResource(baos.toByteArray());

    return Optional.of(resource);
  }

  public Optional<List<ByteArrayResource>> loadFiles(RequestAlbumDTO request) throws IOException {
    Long groupId = request.getGroupId();
    String albumTitle = request.getTitle();

    Optional<TravelGroup> travelGroup = travelGroupService.findGroupById(groupId);

    if (travelGroup.isEmpty()) {
      throw new IOException("PhotoService: Travel Group not found");
    }

    try {
      // 1. 폴더의 객체 리스트 가져오기
      String folderPath = groupId + "/" + albumTitle;
      ListObjectsV2Request listObjectsV2Request = new ListObjectsV2Request()
              .withBucketName(bucketName)
              .withPrefix(folderPath)
              .withDelimiter("/");
      ListObjectsV2Result result = s3Client.listObjectsV2(listObjectsV2Request);

      List<ByteArrayResource> photoResources = new ArrayList<>();

      // 2. 파일 읽어서 ByteArrayResource로 변환
      for (S3ObjectSummary objectSummary : result.getObjectSummaries()) {
        S3Object s3Object = s3Client.getObject(bucketName, objectSummary.getKey());
        S3ObjectInputStream inputStream = s3Object.getObjectContent();
        byte[] content = inputStream.readAllBytes(); // Java 9+ API

        // ByteArrayResource로 변환
        ByteArrayResource resource = new ByteArrayResource(content);
        photoResources.add(resource);
      }
      return Optional.of(photoResources);

    } catch (Exception e) {
      return Optional.empty();
    }
  }

  public boolean doesFileExist(String filePath) {
    S3Object s3Object = s3Client.getObject(bucketName, filePath);
    return s3Object.getObjectMetadata().getContentLength() > 0;
  }

  public void deleteFile(String filePath) {
    s3Client.deleteObject(bucketName, filePath);
    photoRepository.deleteByFilePath(filePath);
  }

}
