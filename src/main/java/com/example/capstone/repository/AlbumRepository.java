package com.example.capstone.repository;

import com.example.capstone.entity.Album;
import com.example.capstone.entity.TravelGroup;
import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

public interface AlbumRepository extends JpaRepository<Album, Long> {

  List<Album> findAllByGroupId(final Long groupId);
  List<Album> findAllByGroup(final TravelGroup group);

  Optional<Album> findByGroupAndTitle(TravelGroup group, String title);
}