package com.example.capstone.controller;

import com.example.capstone.dto.LoginUserDTO;
import com.example.capstone.dto.ResponseUserDTO;
import com.example.capstone.dto.SignupUserDTO;
import com.example.capstone.service.UserService;
import jakarta.validation.Valid;
import org.apache.coyote.BadRequestException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.crossstore.ChangeSetPersister.NotFoundException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/api/users")
public class UserController {

  @Autowired
  private UserService userService;

  // 회원가입
  @PostMapping("/signup")
  public ResponseEntity<Void> signup(@RequestBody @Valid SignupUserDTO user) {
    userService.createUser(user);
    return new ResponseEntity<>(HttpStatus.CREATED);
  }

  // 로그인
  @PostMapping("/login")
  public ResponseEntity<ResponseUserDTO> login(@RequestBody @Valid LoginUserDTO user)
      throws NotFoundException, BadRequestException {
    ResponseUserDTO validatedUser = userService.validateUser(user);

    return new ResponseEntity<>(validatedUser, HttpStatus.OK);
  }

  @ExceptionHandler({MethodArgumentNotValidException.class, NotFoundException.class, BadRequestException.class})
  ResponseEntity<String> handleBadSignupRequest(Exception e) {
    return ResponseEntity.badRequest().body(e.getMessage());
  }
}
