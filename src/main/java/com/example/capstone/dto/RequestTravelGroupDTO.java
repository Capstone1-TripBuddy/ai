package com.example.capstone.dto;

import com.example.capstone.entity.TravelGroup;
import com.example.capstone.entity.User;
import jakarta.validation.constraints.NotBlank;
import java.time.Instant;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Setter
public class RequestTravelGroupDTO {

  @NotBlank
  String groupName;

  @NotBlank
  long creatorId;

  private RequestTravelGroupDTO(String groupName, long creatorId) {
    this.groupName = groupName;
    this.creatorId = creatorId;
  }

  public static RequestTravelGroupDTO createRequestTravelGroupDTO(String groupName, long creatorId) {
    return new RequestTravelGroupDTO(groupName, creatorId);
  }

  // DTO를 Entity로 변환
  public TravelGroup toEntity(final User creator, final String inviteCode, final Instant createdAt) {
    return new TravelGroup(
        this.groupName,
        creator,
        inviteCode,
        createdAt
    );
  }
}
