package com.example.capstone.dto;

import com.example.capstone.entity.User;
import jakarta.validation.constraints.NotBlank;
import lombok.*;
import org.springframework.web.multipart.MultipartFile;

@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Setter
public class RequestSignupUserDTO {

  @NotBlank
  String name;

  @NotBlank
  String email;

  @NotBlank
  String password;

  @Setter
  private MultipartFile profilePicture;


  // DTO를 Entity로 변환
  public static User toEntity(final RequestSignupUserDTO user) {
    return new User(
        user.getEmail(),
        user.getPassword(),
        user.getName());
  }

}
