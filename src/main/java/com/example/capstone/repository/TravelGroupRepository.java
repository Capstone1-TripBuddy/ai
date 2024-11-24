package com.example.capstone.repository;

import com.example.capstone.entity.TravelGroup;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TravelGroupRepository extends JpaRepository<TravelGroup, Long> {

  Optional<TravelGroup> findByInviteCode(String inviteCode);
}
