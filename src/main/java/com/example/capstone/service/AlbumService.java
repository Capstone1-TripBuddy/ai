package com.example.capstone.service;

import com.example.capstone.dto.RequestAlbumDTO;
import com.example.capstone.dto.ResponseAlbumDTO;
import com.example.capstone.dto.ResponsePhotoDTO;
import com.example.capstone.entity.Album;
import com.example.capstone.entity.AlbumPhoto;
import com.example.capstone.entity.Photo;
import com.example.capstone.entity.TravelGroup;
import com.example.capstone.repository.AlbumPhotoRepository;
import com.example.capstone.repository.AlbumRepository;
import com.example.capstone.repository.PhotoRepository;
import com.example.capstone.repository.TravelGroupRepository;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

@Service
public class AlbumService {

  private final TravelGroupRepository travelGroupRepository;
  private final  AlbumRepository albumRepository;
  private final AlbumPhotoRepository albumPhotoRepository;
  private final PhotoRepository photoRepository;
  private final FileService fileService;

  public AlbumService(
      final TravelGroupRepository travelGroupRepository,
      final AlbumRepository albumRepository,
      final AlbumPhotoRepository albumPhotoRepository,
      final PhotoRepository photoRepository,
      final FileService fileService
      ) {
    this.travelGroupRepository = travelGroupRepository;
    this.albumRepository = albumRepository;
    this.albumPhotoRepository = albumPhotoRepository;
    this.photoRepository = photoRepository;
    this.fileService = fileService;
  }


  public List<ResponseAlbumDTO> findAllAlbums(long groupId) {
    Optional<TravelGroup> group = travelGroupRepository.findById(groupId);
    if (group.isEmpty()) {
      return Collections.emptyList();
    }
    List<Album> albums = albumRepository.findAllByGroupId(group.get());

    return albums.stream()
        .map(ResponseAlbumDTO::fromEntity)
        .collect(Collectors.toList());
  }

  public Optional<List<ResponsePhotoDTO>> findAllAlbumPhotos(RequestAlbumDTO request) {
    List<AlbumPhoto> albumPhotos = albumPhotoRepository.findByAlbumTitle(request.getTitle());
    if (albumPhotos.isEmpty()) {
      return Optional.empty();
    }

    // 페이지네이션 및 이미지 정보 매핑
    List<ResponsePhotoDTO> response = new ArrayList<>();
    for (AlbumPhoto albumPhoto : albumPhotos) {
      Photo photo = albumPhoto.getPhoto();

      ResponsePhotoDTO tmp = new ResponsePhotoDTO(
          photo.getFileName(),
          fileService.getFileUrl(photo.getFilePath()),
          photo.getImageSize(),
          photo.getUploadDate()
      );
      response.add(tmp);
    }

    return Optional.of(response);
  }

  public Optional<ResponseEntity<ByteArrayResource>> zipAlbum(RequestAlbumDTO request) throws IOException {
    List<AlbumPhoto> albumPhotos = albumPhotoRepository.findByAlbumTitle(request.getTitle());
    if (albumPhotos.isEmpty()) {
      return Optional.empty();
    }

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ZipOutputStream zipOut = new ZipOutputStream(baos);

    for (AlbumPhoto albumPhoto : albumPhotos) {
      Photo photo = albumPhoto.getPhoto();
      Optional<ByteArrayResource> resource = fileService.loadFile(photo.getFilePath());
      if (resource.isPresent()) {
        ZipEntry zipEntry = new ZipEntry(photo.getFileName()); // 파일 이름 설정 (필요에 따라 수정)
        zipOut.putNextEntry(zipEntry);
        zipOut.write(resource.get().getByteArray());
        zipOut.closeEntry();
      }
    }
    zipOut.close();

    HttpHeaders headers = new HttpHeaders();
    headers.add("Content-Disposition", "attachment; filename=images.zip");
    headers.add("Content-Type", "application/zip");

    return Optional.of(ResponseEntity.ok()
        .headers(headers)
        .body(new ByteArrayResource(baos.toByteArray())));
  }

}
