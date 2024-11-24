package com.example.capstone.service;

import com.example.capstone.dto.LoginUserDTO;
import com.example.capstone.dto.ResponseUserDTO;
import com.example.capstone.dto.SignupUserDTO;
import com.example.capstone.entity.User;
import com.example.capstone.repository.UserRepository;
import org.apache.coyote.BadRequestException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.crossstore.ChangeSetPersister.NotFoundException;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserService {

  @Autowired
  private UserRepository userRepository;

  // Create a new user
  public void createUser(SignupUserDTO user) {
    User createdUser = SignupUserDTO.toEntity(user);
    userRepository.save(createdUser);
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
        foundUser.getProfilePicture()
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

