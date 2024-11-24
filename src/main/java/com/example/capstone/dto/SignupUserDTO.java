package com.example.capstone.dto;

import com.example.capstone.entity.User;
import jakarta.validation.constraints.NotBlank;
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
public class SignupUserDTO {

  @NotBlank
  String name;

  @NotBlank
  String email;

  @NotBlank
  String password;

  // Entity를 DTO로 변환
  public static SignupUserDTO fromEntity(User user) {
    return new SignupUserDTO(
        user.getName(),
        user.getEmail(),
        user.getPassword()
    );
  }

  // DTO를 Entity로 변환
  public static User toEntity(final SignupUserDTO user) {
    return new User(
        user.getEmail(),
        user.getName(),
        null,
        Instant.now());
  }

}
