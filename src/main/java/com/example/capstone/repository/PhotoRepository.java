package com.example.capstone.repository;

import com.example.capstone.entity.Photo;
import com.example.capstone.entity.TravelGroup;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface PhotoRepository extends JpaRepository<Photo, Long> {

  void deleteByFilePath(String filePath);

  List<Photo> findAllByGroup(TravelGroup group);
  List<Photo> findAllByGroupAndAnalyzedAtIsNotNull(TravelGroup group);
  List<Photo> findAllByGroupAndAnalyzedAtIsNull(TravelGroup group);
  List<Photo> findAllByGroupAndPhotoTypeIsNull(TravelGroup group);
}
