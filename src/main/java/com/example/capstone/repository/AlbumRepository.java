package com.example.capstone.repository;

import com.example.capstone.entity.Album;
import com.example.capstone.entity.TravelGroup;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AlbumRepository extends JpaRepository<Album, Long> {

  Album findByTitle(String title);

  List<Album> findAllByGroupId(final TravelGroup group);
}
