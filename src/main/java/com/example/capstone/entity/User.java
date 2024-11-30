package com.example.capstone.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;
import java.time.LocalDateTime;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.DynamicInsert;

@DynamicInsert
@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Entity
@Table(name = "user", schema = "minbak_db")
public class User {

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)  // ID를 자동으로 생성
  private Long id;

  @Column(name = "email", nullable = false)
  private String email;

  @Column(name = "password", nullable = false)
  private String password;

  @Column(name = "name", nullable = false)
  private String name;

  @Setter
  @Column(name = "profile_picture")
  private String profilePicture;

  @ColumnDefault("CURRENT_TIMESTAMP")
  @Column(name = "created_at")
  private LocalDateTime createdAt;

  public User(final String email, final String password, final String name) {
    this.email = email;
    this.password = password;
    this.name = name;
  }
}