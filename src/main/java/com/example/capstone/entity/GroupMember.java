package com.example.capstone.entity;

import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.MapsId;
import jakarta.persistence.Table;
import java.time.Instant;
import java.time.LocalDateTime;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.OnDelete;
import org.hibernate.annotations.OnDeleteAction;

@DynamicInsert
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Entity
@Table(name = "group_member", schema = "minbak_db")
public class GroupMember {

  @EmbeddedId
  private GroupMemberId id;

  @MapsId("groupId")
  @ManyToOne(fetch = FetchType.EAGER, optional = false)
  @OnDelete(action = OnDeleteAction.CASCADE)
  @JoinColumn(name = "group_id", nullable = false)
  private TravelGroup group;

  @MapsId("userId")
  @ManyToOne(fetch = FetchType.EAGER, optional = false)
  @OnDelete(action = OnDeleteAction.CASCADE)
  @JoinColumn(name = "user_id", nullable = false)
  private User user;

  @ColumnDefault("CURRENT_TIMESTAMP")
  @Column(name = "joined_at")
  private LocalDateTime joinedAt;

  public GroupMember(final TravelGroup travelGroup, final User user) {
    this.id = new GroupMemberId(travelGroup.getId(), user.getId());
    this.group = travelGroup;
    this.user = user;
  }
}