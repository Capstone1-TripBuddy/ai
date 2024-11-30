package com.example.capstone.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.Hibernate;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Setter
@Embeddable
public class AlbumPhotoId implements java.io.Serializable {

  private static final long serialVersionUID = -1284548639059197379L;
  @Column(name = "album_id", nullable = false)
  private Long albumId;

  @Column(name = "photo_id", nullable = false)
  private Long photoId;

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || Hibernate.getClass(this) != Hibernate.getClass(o)) {
      return false;
    }
    AlbumPhotoId entity = (AlbumPhotoId) o;
    return Objects.equals(this.albumId, entity.albumId) &&
            Objects.equals(this.photoId, entity.photoId);
  }

  AlbumPhotoId(Long albumId, Long photoId) {
    this.albumId = albumId;
    this.photoId = photoId;
  }

  @Override
  public int hashCode() {
    return Objects.hash(albumId, photoId);
  }

}