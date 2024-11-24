package com.example.capstone.dto;

import com.example.capstone.entity.User;
import jakarta.validation.constraints.NotBlank;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Setter
public class LoginUserDTO {

  @NotBlank
  String email;

  @NotBlank
  String password;

  // Entity를 DTO로 변환
  public static LoginUserDTO fromEntity(User user) {
    return new LoginUserDTO(
        user.getEmail(),
        user.getPassword()
    );
  }

}
