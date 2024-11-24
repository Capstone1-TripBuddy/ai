package com.example.capstone;

import com.example.capstone.controller.UserController;
import com.example.capstone.dto.LoginUserDTO;
import com.example.capstone.dto.ResponseUserDTO;
import com.example.capstone.dto.SignupUserDTO;
import com.example.capstone.service.UserService;
import java.time.Instant;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;

@WebMvcTest(UserController.class)
class UserTest {

  @Autowired
  private MockMvc mockMvc;

  @MockBean
  private UserService userService;

  private SignupUserDTO signupUserDTO;
  private LoginUserDTO loginUserDTO;
  private ResponseUserDTO responseUserDTO;

  @BeforeEach
  void setUp() {
    signupUserDTO = new SignupUserDTO("testUser", "password123", "test@example.com");
    loginUserDTO = new LoginUserDTO("testUser", "password123");
    responseUserDTO = new ResponseUserDTO(12345, "testUser", null);

    MockitoAnnotations.openMocks(this);
  }

  @Test
  @DisplayName("회원가입 성공")
  void testSignup() throws Exception {
    doNothing().when(userService).createUser(any(SignupUserDTO.class));

    mockMvc.perform(MockMvcRequestBuilders.post("/api/users/signup")
            .contentType(MediaType.APPLICATION_JSON)
            .content("{\"name\":\"testUser\", \"password\":\"password123\", \"email\":\"test@example.com\"}"))
        .andExpect(status().isCreated());
  }

  @Test
  @DisplayName("로그인 성공")
  void testLogin() throws Exception {
    when(userService.validateUser(any(LoginUserDTO.class))).thenReturn(responseUserDTO);

    mockMvc.perform(MockMvcRequestBuilders.post("/api/users/login")
            .contentType(MediaType.APPLICATION_JSON)
            .content("{\"email\":\"test@example\", \"password\":\"password123\"}"))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.userId").value(12345))
        .andExpect(jsonPath("$.name").value("testUser"));
  }
}
