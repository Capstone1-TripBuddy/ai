package com.example.capstone;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.example.capstone.controller.TravelGroupController;
import com.example.capstone.dto.RequestGroupMemberDTO;
import com.example.capstone.dto.RequestTravelGroupDTO;
import com.example.capstone.dto.ResponseTravelGroupDTO;
import com.example.capstone.service.TravelGroupService;
import com.example.capstone.service.UserService;
import java.time.Instant;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

class TravelGroupTest {

  @Mock
  private TravelGroupService travelGroupService;

  @Mock
  private UserService userService;

  @InjectMocks
  private TravelGroupController travelGroupController;

  private RequestTravelGroupDTO requestDTO;
  private ResponseTravelGroupDTO responseDTO;
  private RequestGroupMemberDTO memberDTO;
  private Instant now;

  @BeforeEach
  void setUp() {
    now = Instant.now();
    requestDTO = RequestTravelGroupDTO.createRequestTravelGroupDTO("testGroup", 1L);
    responseDTO = ResponseTravelGroupDTO.createResponseTravelGroupDTO(1L, "testGroup", now, "testInviteCode");
    memberDTO = RequestGroupMemberDTO.createResponseTravelGroupDTO(1L, "testInviteCode");

    MockitoAnnotations.openMocks(this);
  }

  @Test
  void testCreateGroupSuccess() {
    when(travelGroupService.createGroup(requestDTO)).thenReturn(Optional.of(responseDTO));

    ResponseEntity<ResponseTravelGroupDTO> response = travelGroupController.createGroup(requestDTO);
    assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);
    verify(travelGroupService, times(1)).createGroup(requestDTO);
  }

  @Test
  void testCreateGroupFailure() {
    when(travelGroupService.createGroup(requestDTO)).thenReturn(null);

    ResponseEntity<ResponseTravelGroupDTO> response = travelGroupController.createGroup(requestDTO);
    assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
  }

  @Test
  void testJoinGroupSuccess() {
    memberDTO.setInviteCode("testInviteCode");
    memberDTO.setUserId(1L);

    when(travelGroupService.joinGroupByInviteCode(memberDTO)).thenReturn(Optional.of(responseDTO));

    ResponseEntity<ResponseTravelGroupDTO> response = travelGroupController.joinGroup(memberDTO);
    assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);
  }

  @Test
  void testJoinGroupFailure() {
    memberDTO.setInviteCode("invalidCode");
    memberDTO.setUserId(1L);

    when(travelGroupService.joinGroupByInviteCode(memberDTO)).thenReturn(Optional.empty());

    ResponseEntity<ResponseTravelGroupDTO> response = travelGroupController.joinGroup(memberDTO);
    assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
  }

  @Test
  void testGetGroupById() {
    Long groupId = 1L;
    when(travelGroupService.getGroupById(groupId)).thenReturn(Optional.of(responseDTO));

    Optional<ResponseTravelGroupDTO> result = travelGroupService.getGroupById(groupId);
    assertThat(result).isPresent();
    assertThat(result.get()).isEqualTo(responseDTO);
    verify(travelGroupService, times(1)).getGroupById(groupId);
  }

  @Test
  void testDeleteGroupSuccess() {
    Long groupId = 1L;
    when(travelGroupService.deleteGroup(groupId)).thenReturn(true);

    boolean isDeleted = travelGroupService.deleteGroup(groupId);
    assertThat(isDeleted).isTrue();
    verify(travelGroupService, times(1)).deleteGroup(groupId);
  }

  @Test
  void testDeleteGroupFailure() {
    Long groupId = 1L;
    when(travelGroupService.deleteGroup(groupId)).thenReturn(false);

    boolean isDeleted = travelGroupService.deleteGroup(groupId);
    assertThat(isDeleted).isFalse();
    verify(travelGroupService, times(1)).deleteGroup(groupId);
  }

  @Nested
  @WebMvcTest(TravelGroupController.class)
  public class TravelGroupApiTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private TravelGroupService travelGroupService;

    @MockBean
    private UserService userService;

    private RequestTravelGroupDTO requestTravelGroupDTO;
    private ResponseTravelGroupDTO responseTravelGroupDTO;
    private RequestGroupMemberDTO requestGroupMemberDTO;

    @BeforeEach
    void setUp() {
      MockitoAnnotations.openMocks(this);
      requestTravelGroupDTO = RequestTravelGroupDTO.createRequestTravelGroupDTO("testGroup", 1L);
      responseTravelGroupDTO = ResponseTravelGroupDTO.createResponseTravelGroupDTO(1L, "testGroup", Instant.now(), "testInviteCode");
      requestGroupMemberDTO = RequestGroupMemberDTO.createResponseTravelGroupDTO(1L, "testInviteCode");
    }

    @Test
    @DisplayName("Create Group - Success")
    void testCreateGroupSuccess() throws Exception {
      when(travelGroupService.createGroup(any(RequestTravelGroupDTO.class))).thenReturn(Optional.of(responseTravelGroupDTO));

      mockMvc.perform(MockMvcRequestBuilders.post("/api/groups")
              .contentType(MediaType.APPLICATION_JSON)
              .content("{\"name\":\"testGroup\", \"creatorId\":1}"))
          .andExpect(status().isCreated())
          .andExpect(jsonPath("$.id").value(1L))
          .andExpect(jsonPath("$.groupName").value("testGroup"))
          .andExpect(jsonPath("$.inviteCode").value("testInviteCode"));
    }

    @Test
    @DisplayName("Create Group - Failure")
    void testCreateGroupFailure() throws Exception {
      when(travelGroupService.createGroup(any(RequestTravelGroupDTO.class))).thenReturn(Optional.empty());

      mockMvc.perform(MockMvcRequestBuilders.post("/api/groups")
              .contentType(MediaType.APPLICATION_JSON)
              .content("{\"name\":\"testGroup\", \"creatorId\":1}"))
          .andExpect(status().isNotFound());
    }

    @Test
    @DisplayName("Join Group - Success")
    void testJoinGroupSuccess() throws Exception {
      when(travelGroupService.joinGroupByInviteCode(any(RequestGroupMemberDTO.class))).thenReturn(Optional.of(responseTravelGroupDTO));

      mockMvc.perform(MockMvcRequestBuilders.post("/api/groups/members")
              .contentType(MediaType.APPLICATION_JSON)
              .content("{\"userId\":1, \"inviteCode\":\"testInviteCode\"}"))
          .andExpect(status().isCreated())
          .andExpect(jsonPath("$.id").value(1L))
          .andExpect(jsonPath("$.groupName").value("testGroup"))
          .andExpect(jsonPath("$.inviteCode").value("testInviteCode"));
    }

    @Test
    @DisplayName("Join Group - Failure")
    void testJoinGroupFailure() throws Exception {
      when(travelGroupService.joinGroupByInviteCode(any(RequestGroupMemberDTO.class))).thenReturn(Optional.empty());

      mockMvc.perform(MockMvcRequestBuilders.post("/api/groups/members")
              .contentType(MediaType.APPLICATION_JSON)
              .content("{\"userId\":1, \"inviteCode\":\"invalidCode\"}"))
          .andExpect(status().isBadRequest());
    }

    @Test
    @DisplayName("Get Group By Id - Success")
    void testGetGroupById() throws Exception {
      when(travelGroupService.getGroupById(1L)).thenReturn(Optional.of(responseTravelGroupDTO));

      mockMvc.perform(MockMvcRequestBuilders.get("/api/groups/1"))
          .andExpect(status().isOk())
          .andExpect(jsonPath("$.id").value(1L))
          .andExpect(jsonPath("$.groupName").value("testGroup"))
          .andExpect(jsonPath("$.inviteCode").value("testInviteCode"));
    }

    @Test
    @DisplayName("Get Group By Id - Failure")
    void testGetGroupByIdFailure() throws Exception {
      when(travelGroupService.getGroupById(1L)).thenReturn(Optional.empty());

      mockMvc.perform(MockMvcRequestBuilders.get("/api/groups/1"))
          .andExpect(status().isNotFound());
    }

    @Test
    @DisplayName("Delete Group - Success")
    void testDeleteGroupSuccess() throws Exception {
      when(travelGroupService.deleteGroup(1L)).thenReturn(true);

      mockMvc.perform(MockMvcRequestBuilders.delete("/api/groups/1"))
          .andExpect(status().isOk());
    }

    @Test
    @DisplayName("Delete Group - Failure")
    void testDeleteGroupFailure() throws Exception {
      when(travelGroupService.deleteGroup(1L)).thenReturn(false);

      mockMvc.perform(MockMvcRequestBuilders.delete("/api/groups/1"))
          .andExpect(status().isBadRequest());
    }
  }
}
