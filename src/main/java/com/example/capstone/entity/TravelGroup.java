package com.example.capstone.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.OnDelete;
import org.hibernate.annotations.OnDeleteAction;

@DynamicInsert
@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Entity
@Table(name = "travel_group", schema = "minbak_db")
public class TravelGroup {

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)  // ID를 자동으로 생성
  private Long id;

  @Column(name = "group_name", nullable = false)
  private String groupName;

  @ManyToOne(fetch = FetchType.EAGER, optional = false)
  @OnDelete(action = OnDeleteAction.CASCADE)
  @JoinColumn(name = "creator_id", nullable = false)
  private User creator;

  @Column(name = "invite_code", nullable = false)
  private String inviteCode;

  @ColumnDefault("CURRENT_TIMESTAMP")
  @Column(name = "created_at")
  private LocalDateTime createdAt;

  public TravelGroup(final String groupName, final User creator, final String inviteCode) {
    this.groupName = groupName;
    this.creator = creator;
    this.inviteCode = inviteCode;
  }
}