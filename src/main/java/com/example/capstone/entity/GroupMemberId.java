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
public class GroupMemberId implements java.io.Serializable {

  private static final long serialVersionUID = -3416136616810012856L;
  @Column(name = "group_id", nullable = false)
  private Long groupId;

  @Column(name = "user_id", nullable = false)
  private Long userId;

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || Hibernate.getClass(this) != Hibernate.getClass(o)) {
      return false;
    }
    GroupMemberId entity = (GroupMemberId) o;
    return Objects.equals(this.groupId, entity.groupId) &&
            Objects.equals(this.userId, entity.userId);
  }

  GroupMemberId(final Long groupId, final Long userId) {
    this.groupId = groupId;
    this.userId = userId;
  }

  @Override
  public int hashCode() {
    return Objects.hash(groupId, userId);
  }

}