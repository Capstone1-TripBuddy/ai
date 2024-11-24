package com.example.capstone.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import java.util.Objects;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.Hibernate;

@Getter
@Setter
@Embeddable
public class UserFaceId implements java.io.Serializable {

  private static final long serialVersionUID = -4822374499376806705L;
  @Column(name = "user_id", nullable = false)
  private Long userId;

  @Column(name = "face_id", nullable = false)
  private Long faceId;

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || Hibernate.getClass(this) != Hibernate.getClass(o)) {
      return false;
    }
    UserFaceId entity = (UserFaceId) o;
    return Objects.equals(this.faceId, entity.faceId) &&
        Objects.equals(this.userId, entity.userId);
  }

  @Override
  public int hashCode() {
    return Objects.hash(faceId, userId);
  }

}