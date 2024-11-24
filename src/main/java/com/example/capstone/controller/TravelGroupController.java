package com.example.capstone.controller;

import com.example.capstone.dto.RequestGroupMemberDTO;
import com.example.capstone.dto.RequestTravelGroupDTO;
import com.example.capstone.dto.ResponseTravelGroupDTO;
import com.example.capstone.service.TravelGroupService;
import com.example.capstone.service.UserService;
import java.util.Optional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/groups")
public class TravelGroupController {

  @Autowired
  private TravelGroupService travelGroupService;

  @Autowired
  private UserService userService;

  // Create a new travel group
  @PostMapping("/")
  public ResponseEntity<ResponseTravelGroupDTO> createGroup(@RequestBody RequestTravelGroupDTO travelGroupDTO) {
    Optional<ResponseTravelGroupDTO> createdGroup = travelGroupService.createGroup(travelGroupDTO);
    return createdGroup.map(
            responseTravelGroupDTO -> new ResponseEntity<>(responseTravelGroupDTO, HttpStatus.CREATED))
        .orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
  }

  @PostMapping("/members")
  public ResponseEntity<ResponseTravelGroupDTO> joinGroup(@RequestBody RequestGroupMemberDTO requestGroupMemberDTO) {
    Optional<ResponseTravelGroupDTO> travelGroupDTO = travelGroupService.joinGroupByInviteCode(requestGroupMemberDTO);

    return travelGroupDTO.map(
            responseTravelGroupDTO -> new ResponseEntity<>(responseTravelGroupDTO, HttpStatus.CREATED))
        .orElseGet(() -> new ResponseEntity<>(HttpStatus.BAD_REQUEST));
  }
}