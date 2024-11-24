package com.example.capstone.controller;

import com.example.capstone.dto.RequestPhotoDTO;
import com.example.capstone.service.PhotoService;
import com.example.capstone.service.S3Service;
import java.io.IOException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("api/photos")
public class PhotoController {

  private final S3Service s3Service;
  private final PhotoService photoService;

  public PhotoController(S3Service s3Service, PhotoService photoService) {
    this.s3Service = s3Service;
    this.photoService = photoService;
  }



  @ExceptionHandler
  public ResponseEntity<String> handleException(Exception ex) {
    return new ResponseEntity<>(ex.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
  }
}


