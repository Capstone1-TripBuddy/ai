package com.example.capstone.dto;

import com.example.capstone.entity.Album;
import com.example.capstone.entity.User;
import java.time.Instant;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Setter
public class ResponseAlbumDTO {

  private long id;
  private String title;
  private User creator;
  private Instant createdAt;
  private String description;

  public static ResponseAlbumDTO fromEntity(Album album) {
    if (album == null) {
      throw new IllegalArgumentException("Album cannot be null");
    }

    // 혹시 Album 생성자중 id가 없는 생성자가 존재하기 때문?
    return new ResponseAlbumDTO(
        album.getId(),
        album.getTitle(),
        album.getUser(),
        album.getCreatedAt(),
        album.getDescription()
    );
  }
}
