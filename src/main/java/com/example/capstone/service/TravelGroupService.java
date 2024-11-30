package com.example.capstone.service;

import com.amazonaws.services.s3.model.ExistingObjectReplication;
import com.example.capstone.dto.RequestGroupMemberDTO;
import com.example.capstone.dto.RequestTravelGroupDTO;
import com.example.capstone.dto.ResponseTravelGroupDTO;
import com.example.capstone.entity.GroupMember;
import com.example.capstone.entity.TravelGroup;
import com.example.capstone.entity.User;
import com.example.capstone.repository.GroupMemberRepository;
import com.example.capstone.repository.TravelGroupRepository;
import com.example.capstone.repository.UserRepository;
import jakarta.persistence.EntityExistsException;
import jakarta.persistence.EntityNotFoundException;
import java.time.Instant;
import java.util.Objects;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class TravelGroupService {

  @Autowired
  private GroupMemberRepository groupMemberRepository;

  @Autowired
  private TravelGroupRepository travelGroupRepository;

  @Autowired
  private UserRepository userRepository;


  // Create a new travel group
  public Optional<ResponseTravelGroupDTO> createGroup(RequestTravelGroupDTO travelGroup) {
    Optional<User> creator = userRepository.findById(travelGroup.getCreatorId());
    if (creator.isEmpty()) {
      return Optional.empty();
    }

    // UNIQUE KEY (userId, groupName)
    List<TravelGroup> existingGroups = travelGroupRepository.findAll().stream()
            .filter((group) -> group.getCreator().equals(creator.get()))
            .filter((group) -> Objects.equals(group.getGroupName(), travelGroup.getGroupName())).toList();
    if (!existingGroups.isEmpty()) {
      throw new EntityExistsException();
    }

    TravelGroup createdGroup = travelGroup.toEntity(
            creator.get(),
            this.generateInviteCode(10)
    );
    TravelGroup result = travelGroupRepository.save(createdGroup);

    return Optional.of(ResponseTravelGroupDTO.fromEntity(result));
  }

  // Get all travel groups
  public List<ResponseTravelGroupDTO> getAllGroups() {
    return travelGroupRepository.findAll().stream()
            .map(ResponseTravelGroupDTO::fromEntity)
            .collect(Collectors.toList());
  }

  // Get all travel members
  public List<ResponseTravelGroupDTO> getAllGroups(Long groupId) {
    travelGroupRepository.findById(groupId).ifPresent(travelGroup -> {

    });
    List<GroupMember> groupMembers = groupMemberRepository.findByGroupId(groupId);
    if (groupMembers.isEmpty()) {
      throw new EntityNotFoundException("getAllGroups: can't find any group member");
    }

    return travelGroupRepository.findAll().stream()
            .map(ResponseTravelGroupDTO::fromEntity)
            .collect(Collectors.toList());
  }

  // Join travel group by invite code
  public Optional<ResponseTravelGroupDTO> joinGroupByInviteCode(RequestGroupMemberDTO groupMemberDTO) {
    Optional<TravelGroup> travelGroup = travelGroupRepository.findByInviteCode(groupMemberDTO.getInviteCode());
    Optional<User> user = userRepository.findById(groupMemberDTO.getUserId());

    if (travelGroup.isPresent() && user.isPresent()) {
      GroupMember createdGroupMember = new GroupMember(travelGroup.get(), user.get());
      groupMemberRepository.save(createdGroupMember);

      return Optional.of(ResponseTravelGroupDTO.fromEntity(travelGroup.get()));
    }
    return Optional.empty();
  }

  // Delete travel group by ID
  public boolean deleteGroup(Long id) {
    if (travelGroupRepository.existsById(id)) {
      travelGroupRepository.deleteById(id);
      return true;
    }
    return false;
  }

  // Generate a unique invite code
  private String generateInviteCode(int length) {
    return UUID.randomUUID().toString().replace("-", "").substring(0, length);
  }

  public Optional<TravelGroup> findGroupById(final Long id) {
    return travelGroupRepository.findById(id);
  }
}
