package com.example.capstone.service;

import com.amazonaws.HttpMethod;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.GeneratePresignedUrlRequest;
import com.example.capstone.dto.LoginUserDTO;
import com.example.capstone.dto.RequestSignupUserDTO;
import com.example.capstone.dto.ResponseUserDTO;
import com.example.capstone.entity.User;
import com.example.capstone.repository.UserRepository;
import java.io.IOException;
import org.apache.coyote.BadRequestException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.crossstore.ChangeSetPersister.NotFoundException;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.net.URL;
import java.util.Date;
import java.util.List;
import java.util.Optional;

@Service
public class UserService {

  @Value("${cloud.aws.s3.bucket}")
  private String bucketName;

  @Autowired
  private UserRepository userRepository;

  @Autowired
  private AmazonS3Client s3Client;
  // Create a new user
  public Optional<User> createUser(RequestSignupUserDTO user) throws IOException {
    User createdUser = RequestSignupUserDTO.toEntity(user);
    userRepository.save(createdUser);

    return Optional.of(createdUser);
  }

  public String generateSignedUrl(String filePath) {
    try {
      // Signed URL 요청 생성
      int duration = 60 * 60;
      GeneratePresignedUrlRequest generatePresignedUrlRequest =
              new GeneratePresignedUrlRequest(bucketName, filePath)
                      .withMethod(HttpMethod.GET) // HTTP GET 요청용 Signed URL
                      .withExpiration(new Date(System.currentTimeMillis() + duration * 1000));

      // Signed URL 생성
      URL signedUrl = s3Client.generatePresignedUrl(generatePresignedUrlRequest);
      return signedUrl.toString();

    } catch (Exception e) {
      throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR,
              "Failed to generate signed URL: " + e.getMessage(), e);
    }
  }

  // Validate a user
  public ResponseUserDTO validateUser(LoginUserDTO user) throws NotFoundException, BadRequestException {
    User foundUser = userRepository.findByEmail(user.getEmail());
    if (foundUser == null) {
      throw new NotFoundException();
    }
    if (!foundUser.getPassword().equals(user.getPassword())) {
      throw new BadRequestException();
    }

    return new ResponseUserDTO(
            foundUser.getId(),
            foundUser.getName(),
            generateSignedUrl(foundUser.getProfilePicture())
    );
  }

  // Get user by ID
  public Optional<User> getUserById(Long id) {
    return userRepository.findById(id);
  }

  // Get user by Email
  public User getUserByEmail(final String email) {
    return userRepository.findByEmail(email);
  }

  // Get all users
  public List<User> getAllUsers() {
    return userRepository.findAll();
  }

  // Update user details
  public Optional<User> updateUser(Long id, User updatedUser) {
    return userRepository.findById(id).map(existingUser -> {
      User updated = new User(
              existingUser.getId(),  // ID는 변경되지 않음
              updatedUser.getEmail(),
              updatedUser.getPassword(),
              updatedUser.getName(),
              updatedUser.getProfilePicture(),
              existingUser.getCreatedAt() // created_at 필드는 변경하지 않음
      );
      return userRepository.save(updated);
    });
  }

  // Delete user by ID
  public boolean deleteUser(Long id) {
    if (userRepository.existsById(id)) {
      userRepository.deleteById(id);
      return true;
    }
    return false;
  }

  public Optional<User> findUserById(final Long id) {
    return userRepository.findById(id);
  }
}

