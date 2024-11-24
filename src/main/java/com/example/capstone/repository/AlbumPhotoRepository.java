package com.example.capstone.repository;

import com.example.capstone.entity.AlbumPhoto;
import com.example.capstone.entity.AlbumPhotoId;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AlbumPhotoRepository extends JpaRepository<AlbumPhoto, AlbumPhotoId> {

  List<AlbumPhoto> findByAlbumId(Long albumId);
  List<AlbumPhoto> findByAlbumTitle(String albumTitle);
}
