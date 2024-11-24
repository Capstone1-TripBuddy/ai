package com.example.capstone.repository;

import com.example.capstone.entity.GroupMember;
import com.example.capstone.entity.TravelGroup;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface GroupMemberRepository extends JpaRepository<GroupMember, Long> {

    List<GroupMember> findAllByGroup(TravelGroup group);
}
