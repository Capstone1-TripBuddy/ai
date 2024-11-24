package com.example.capstone.repository;

import com.example.capstone.entity.UserFace;
import com.example.capstone.entity.UserFaceId;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserFaceRepository extends JpaRepository<UserFace, UserFaceId> {
}
