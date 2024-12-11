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

  Long userId;

  String inviteCode;


  private RequestGroupMemberDTO(Long userId, String inviteCode) {
    this.userId = userId;
    this.inviteCode = inviteCode;
  }

  public static RequestGroupMemberDTO createResponseTravelGroupDTO(Long userId, String inviteCode) {
    return new RequestGroupMemberDTO(userId, inviteCode);
  }

}
