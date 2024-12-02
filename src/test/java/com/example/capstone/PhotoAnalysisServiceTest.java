package com.example.capstone;

import com.example.capstone.repository.GroupMemberRepository;
import com.example.capstone.repository.PhotoRepository;
import com.example.capstone.repository.TravelGroupRepository;
import com.example.capstone.repository.UserRepository;
import com.example.capstone.service.PhotoAnalysisService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

@SpringBootTest
public class PhotoAnalysisServiceTest {
    @Autowired
    UserRepository userRepository;
    @Autowired
    TravelGroupRepository travelGroupRepository;
    @Autowired
    PhotoRepository photoRepository;
    @Autowired
    GroupMemberRepository groupMemberRepository;
    @Autowired
    PhotoAnalysisService photoAnalysisService;

    @Test
    @Transactional
    public void testIsValidProfileImage() throws IOException {
        MultipartFile multipartFile = new MockMultipartFile("image.jpg", new FileInputStream(new File("C:\\Users\\User\\Pictures\\proj3\\all\\Donald_Trump2.jpg")));
        int a = photoAnalysisService.isValidProfileImage(multipartFile);
        System.out.println(a);
    }

    @Test
    @Transactional
    public void testGetImageQueations() throws IOException {
        MultipartFile multipartFile = new MockMultipartFile("image.jpg", new FileInputStream(new File("C:\\Users\\User\\Pictures\\proj3\\all\\Beach1.jpg")));
        String[] s = photoAnalysisService.getImageQueations(multipartFile);
        for (String string : s) {
            System.out.println(string);
        }
    }

    @Test
    @Transactional
    public void testProcessImagesTypes() {
        photoAnalysisService.processImagesTypes(2);
    }

    @Test
    @Transactional
    public void testProcessImagesFaces() {
        photoAnalysisService.processImagesFaces(2, true);
    }
}
