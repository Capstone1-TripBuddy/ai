package com.example.capstone.controller;

import com.example.capstone.dto.LoginUserDTO;
import com.example.capstone.dto.RequestSignupUserDTO;
import com.example.capstone.dto.ResponseUserDTO;
import com.example.capstone.entity.User;
import com.example.capstone.service.FileService;
import com.example.capstone.service.UserService;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.Optional;
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

  @Autowired
  private FileService fileService;

  // 회원가입
  @PostMapping("/signup")
  public ResponseEntity<Void> signup(@ModelAttribute @Valid RequestSignupUserDTO request)
          throws IOException {
    Optional<User> createdUser = userService.createUser(request);
    if (createdUser.isEmpty()) {
      return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
    }
    fileService.storeProfilePicture(createdUser.get(), request.getProfilePicture());
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
