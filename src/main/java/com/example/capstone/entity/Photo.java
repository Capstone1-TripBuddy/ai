package com.example.capstone.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.Instant;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.OnDelete;
import org.hibernate.annotations.OnDeleteAction;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Entity
@Table(name = "photo", schema = "minbak_db")
public class Photo {

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)  // ID를 자동으로 생성
  private Long id;

  @ManyToOne(fetch = FetchType.LAZY, optional = false)
  @OnDelete(action = OnDeleteAction.CASCADE)
  @JoinColumn(name = "user_id", nullable = false)
  private User user;

  @ManyToOne(fetch = FetchType.LAZY, optional = false)
  @OnDelete(action = OnDeleteAction.CASCADE)
  @JoinColumn(name = "group_id", nullable = false)
  private TravelGroup group;

  @Column(name = "file_name", nullable = false)
  private String fileName;

  @Column(name = "file_path")
  private String filePath;

  @ColumnDefault("CURRENT_TIMESTAMP")
  @Column(name = "upload_date")
  private Instant uploadDate;

  @Column(name = "image_size")
  private Long imageSize;

  @Column(name = "image_format", length = 50)
  private String imageFormat;

  @Lob
  @Column(name = "meta_data")
  private String metaData;

  @Setter
  @Column(name = "photo_type", length = 50)
  private String photoType;

  @Setter
  @ColumnDefault("0")
  @Column(name = "has_face")
  private Boolean hasFace;

  @Setter
  @Column(name = "analyzed_at")
  private Instant analyzedAt;

  public Photo(
      final User user,
      final TravelGroup travelGroup,
      final String filename,
      final String filepath,
      final Long imageSize,
      String imageFormat
      ) {
  }

}