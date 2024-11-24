package com.example.capstone.dto;

import com.example.capstone.entity.TravelGroup;
import java.time.Instant;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Setter
public class ResponseTravelGroupDTO {

  long id;

  String groupName;

  Instant createdAt;

  String inviteCode;

  private ResponseTravelGroupDTO(long id, String groupName, Instant createdAt, String inviteCode) {
    this.id = id;
    this.groupName = groupName;
    this.createdAt = createdAt;
    this.inviteCode = inviteCode;
  }

  public static ResponseTravelGroupDTO createResponseTravelGroupDTO(long id, String groupName, Instant createdAt, String inviteCode) {
    return new ResponseTravelGroupDTO(id, groupName, createdAt, inviteCode);
  }

  // Entity를 DTO로 변환
  public static ResponseTravelGroupDTO fromEntity(TravelGroup travelGroup) {
    return new ResponseTravelGroupDTO(
        travelGroup.getId(),
        travelGroup.getGroupName(),
        travelGroup.getCreatedAt(),
        travelGroup.getInviteCode()
    );
  }

}
