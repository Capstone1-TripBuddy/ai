package com.example.capstone.dto;

import java.time.Instant;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
public class ResponsePhotoDTO {

  private String fileName;
  private String fileUrl;
  private Long imageSize;
  private Instant uploadDate;

}
