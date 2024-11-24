package com.example.capstone.repository;

import com.example.capstone.entity.Face;
import com.example.capstone.entity.TravelGroup;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FaceRepository extends JpaRepository<Face, Long> {
    List<Face> findAllByPhotoGroup(TravelGroup group);
}
