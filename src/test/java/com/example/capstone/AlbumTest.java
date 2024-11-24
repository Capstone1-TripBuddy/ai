package com.example.capstone;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.example.capstone.dto.RequestAlbumDTO;
import com.example.capstone.dto.ResponseAlbumDTO;
import com.example.capstone.entity.Album;
import com.example.capstone.entity.AlbumPhoto;
import com.example.capstone.entity.Photo;
import com.example.capstone.entity.TravelGroup;
import com.example.capstone.entity.User;
import com.example.capstone.repository.AlbumPhotoRepository;
import com.example.capstone.repository.AlbumRepository;
import com.example.capstone.repository.PhotoRepository;
import com.example.capstone.repository.TravelGroupRepository;
import com.example.capstone.repository.UserRepository;
import com.example.capstone.service.AlbumService;
import com.example.capstone.service.PhotoService;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

class AlbumTest {

  @Mock
  private AlbumRepository albumRepository;

  @Mock
  private AlbumPhotoRepository albumPhotoRepository;

  @Mock
  private UserRepository userRepository;

  @Mock
  private TravelGroupRepository travelGroupRepository;

  @Mock
  private PhotoRepository photoRepository;

  @Mock
  private PhotoService photoService;

  @InjectMocks
  private AlbumService albumService;

  private Album album;
  private TravelGroup travelGroup;
  private RequestAlbumDTO requestAlbumDTO;
  private ResponseAlbumDTO responseAlbumDTO;
  private Photo photo;
  private AlbumPhoto albumPhoto;

  @BeforeEach
  void setUp() {
    MockitoAnnotations.openMocks(this);

    // 데이터 셋업
    User user = userRepository.findById(1L).get();
    travelGroup = travelGroupRepository.findById(1L).get();
    album = albumRepository.findById(1L).get();

    requestAlbumDTO = new RequestAlbumDTO(1L, "testAlbum");
    responseAlbumDTO = new ResponseAlbumDTO(
        1L,  // ID가 1L로 설정됨
        "testTitle",
        user,
        Instant.now(),
        "testDescription"
    );

    // photo ID 값 설정
    photo = new Photo(
        1L,  // ID가 1L로 설정됨
        user,
        travelGroup,
        "filename",
        "filepath",
        Instant.now()
    );

    // AlbumPhoto 객체 생성
    albumPhoto = new AlbumPhoto(
        album.getId(),
        photo.getId()
    );
  }

  @Test
  void testFindAllAlbums() {
    // given
    when(albumRepository.findAllByGroupId(travelGroup)).thenReturn(Collections.singletonList(album));

    // when
    List<ResponseAlbumDTO> albums = albumService.findAllAlbums(1L);

    // then
    assertNotNull(albums);
    assertEquals(1, albums.size());
    assertEquals("testAlbum", albums.get(0).getTitle());
    verify(albumRepository, times(1)).findAllByGroupId(travelGroup);
  }

  @Test
  void testFindAllPhotos() {
    // given
    when(albumRepository.findByTitle("testAlbum")).thenReturn(album);
    when(albumPhotoRepository.findByAlbumId(album.getId())).thenReturn(
        Collections.singletonList(albumPhoto));
    when(photoRepository.findById(photo.getId())).thenReturn(Optional.of(photo));
    when(photoService.downloadByFilename(photo.getFilename())).thenReturn(
        Optional.of(new byte[]{1, 2, 3}));

    // when
    List<Optional<byte[]>> photos = albumService.findAllPhotos(requestAlbumDTO);

    // then
    assertNotNull(photos);
    assertEquals(1, photos.size());
    assertTrue(photos.get(0).isPresent());
    verify(albumRepository, times(1)).findByTitle("testAlbum");
    verify(albumPhotoRepository, times(1)).findByAlbumId(album.getId());
    verify(photoRepository, times(1)).findById(photo.getId());
    verify(photoService, times(1)).downloadByFilename(photo.getFilename());
  }

  @Test
  void testFindAllPhotosWithNoPhotos() {
    // given
    when(albumRepository.findByTitle("testAlbum")).thenReturn(album);
    when(albumPhotoRepository.findByAlbumId(album.getId())).thenReturn(List.of());

    // when
    List<Optional<byte[]>> photos = albumService.findAllPhotos(requestAlbumDTO);

    // then
    assertNotNull(photos);
    assertTrue(photos.isEmpty());
    verify(albumRepository, times(1)).findByTitle("testAlbum");
    verify(albumPhotoRepository, times(1)).findByAlbumId(album.getId());
    verify(photoRepository, never()).findById(anyLong());
    verify(photoService, never()).downloadByFilename(anyString());
  }
}
