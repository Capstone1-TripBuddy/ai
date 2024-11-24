package com.example.capstone.dto;

import com.example.capstone.entity.GroupMember;
import com.example.capstone.entity.TravelGroup;
import com.example.capstone.entity.User;
import java.time.Instant;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.web.multipart.MultipartFile;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@Setter
public class RequestGroupMemberDTO {

  long userId;

  String inviteCode;


  private RequestGroupMemberDTO(long userId, String inviteCode) {
    this.userId = userId;
    this.inviteCode = inviteCode;
  }

  public static RequestGroupMemberDTO createResponseTravelGroupDTO(long userId, String inviteCode) {
    return new RequestGroupMemberDTO(userId, inviteCode);
  }

  public static GroupMember toEntity(TravelGroup travelGroup, User user) {
    return new GroupMember(
        travelGroup,
        user
    );
  }

}
