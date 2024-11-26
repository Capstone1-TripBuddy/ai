package com.example.capstone.service;

import com.example.capstone.dto.PhotoAnalysisDto;
import com.example.capstone.dto.PhotoFaceDto;
import com.example.capstone.entity.*;
import com.example.capstone.repository.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.*;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentHashMap;

@Async
@Service
@RequiredArgsConstructor
public class PhotoAnalysisService {
    private final TravelGroupRepository travelGroupRepository;
    private final GroupMemberRepository groupMemberRepository;
    private final PhotoRepository photoRepository;
    private final FaceRepository faceRepository;
    private final UserFaceRepository userFaceRepository;
    private final ConcurrentHashMap<Long, Object> lockMap = new ConcurrentHashMap<>();
    private final String pythonServerUrl = "http://127.0.0.1:8000/process_photos/";

    // groupId에 해당하는 Lock 객체 가져오기
    private Object getLock(Long groupId) {
        return lockMap.computeIfAbsent(groupId, id -> new Object());
    }

    private void removeLockIfUnused(Long groupId, Object lock) {
        lockMap.computeIfPresent(groupId, (id, existingLock) -> {
            if (existingLock == lock && Thread.holdsLock(lock)) {
                return null; // 사용되지 않는 경우 제거
            }
            return existingLock;
        });
    }

    @Async
    public boolean isValidProfileImage(MultipartFile file) {
        try {
            // Convert MultipartFile to byte array
            byte[] fileBytes = file.getBytes();

            // Create headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // Create request entity
            HttpEntity<byte[]> requestEntity = new HttpEntity<>(fileBytes, headers);

            // Send POST request and receive response as PhotoFaceDto[]
            ResponseEntity<PhotoFaceDto[]> response = restTemplate.exchange(
                    "http://localhost:8000/test/faces",
                    HttpMethod.POST,
                    requestEntity,
                    PhotoFaceDto[].class
            );

            // Check response status
            if (response.getStatusCode() != HttpStatus.OK || response.getBody() == null) {
                return false;
            }

            // Validate profile image based on face detection
            PhotoFaceDto[] faceData = response.getBody();
            return faceData.length == 1; // Example: valid if at least one face is detected

        } catch (IOException | IllegalStateException e) {
            e.printStackTrace();
            return false;
        }
    }

    @Async
    @Transactional
    public void processImagesTypes(long groupId) {
        TravelGroup group = travelGroupRepository.findById(groupId).orElseThrow(
                () -> new NoSuchElementException("Travel group not found")
        );
        Object lock = getLock(groupId);

        synchronized (lock) {
            try {
                List<Photo> newPhotoList = photoRepository.findAllByGroupAndPhotoTypeIsNull(group);
                List<String> newPhotoPaths = new ArrayList<>();

                for (Photo photo : newPhotoList) {
                    newPhotoPaths.add(photo.getFilePath());
                }

                RestTemplate restTemplate = new RestTemplate();
                HttpHeaders headers = new HttpHeaders();

                MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                body.add("photo_paths", newPhotoPaths);

                HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

                ResponseEntity<String[]> response = restTemplate.exchange(
                        pythonServerUrl,
                        HttpMethod.POST,
                        requestEntity,
                        String[].class
                );

                String[] categories = response.getBody();

                for (int i = 0, l = categories == null ? 0 : categories.length; i < l; i++) {
                    newPhotoList.get(i).setPhotoType(categories[i]);
                }
                photoRepository.saveAll(newPhotoList);
            } finally {
                // 필요에 따라 Lock 객체를 제거 (메모리 관리)
                removeLockIfUnused(groupId, lock);
            }
        }
    }

    @Async
    @Transactional
    public void processImagesFaces(long groupId) {
        TravelGroup group = travelGroupRepository.findById(groupId).orElseThrow(
                () -> new NoSuchElementException("Travel group not found")
        );
        Object lock = getLock(groupId);

        synchronized (lock) {
            try {
                // 현재 Group의 얼굴 데이터를 가져오기
                List<Face> oldFaceList = faceRepository.findAllByPhotoGroup(group);
                List<Face> newFaceList = new ArrayList<>();
                List<UserFace> newUserFaceList = new ArrayList<>();
                List<Photo> oldPhotoList = photoRepository.findAllByGroupAndAnalyzedAtIsNotNull(group);
                List<Photo> newPhotoList = photoRepository.findAllByGroupAndAnalyzedAtIsNull(group);
                List<String> newPhotoPaths = new ArrayList<>();
                List<GroupMember> groupMemberList = groupMemberRepository.findAllByGroup(group);
                List<String> groupMemberProfilePics = new ArrayList<>();
                List<String> groupMemberNames = new ArrayList<>();

                for (Photo photo : newPhotoList) {
                    newPhotoPaths.add(photo.getFilePath());
                }

                for (int i = 0; i < groupMemberList.size(); ++i) {
                    groupMemberProfilePics.add(groupMemberList.get(i).getUser().getProfilePicture());
                    groupMemberNames.add(String.valueOf(i));
                }

                // 새로 인식된 얼굴 데이터 처리
                RestTemplate restTemplate = new RestTemplate();
                HttpHeaders headers = new HttpHeaders();

                MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                body.add("profile_image_paths", groupMemberProfilePics);
                body.add("profile_names", groupMemberNames);
                body.add("photo_paths", newPhotoPaths);
                //body.add("embedding_ids", embeddingIds.stream().map(String::valueOf).collect(Collectors.joining(",")));

                HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

                ResponseEntity<PhotoFaceDto[][]> response = restTemplate.exchange(
                        pythonServerUrl,
                        HttpMethod.POST,
                        requestEntity,
                        PhotoFaceDto[][].class
                );

                // 데이터 업데이트
                PhotoFaceDto[][] photosData = response.getBody();

                for (int i = 0, lp = photosData == null ? 0 : photosData.length; i < lp; i++) {
                    Photo photo = newPhotoList.get(i);
                    PhotoFaceDto[] facesData = photosData[i];
                    photo.setHasFace(facesData != null && facesData.length > 0);
                    photo.setAnalyzedAt(Instant.now());
                    photoRepository.save(photo);
                    for (int j = 0, lf = facesData == null ? 0 : facesData.length; j < lf; j++) {
                        PhotoFaceDto faceData = facesData[j];
                        if (faceData.getLabel() != null) {
                            String boundingBox = String.format(
                                    "%d,%d,%d,%d",
                                    faceData.getX(), faceData.getY(), faceData.getW(), faceData.getH()
                            );
                            Face face = new Face(photo, boundingBox);
                            faceRepository.save(face);

                            int idx = Integer.getInteger(faceData.getLabel());
                            User user = groupMemberList.get(idx).getUser();
                            userFaceRepository.save(new UserFace(user, face));
                        }
                    }
                }
            } finally {
                // 필요에 따라 Lock 객체를 제거 (메모리 관리)
                removeLockIfUnused(groupId, lock);
            }
        }
    }
}
