package com.example.capstone.controller;

import com.example.capstone.dto.RequestAlbumDTO;
import com.example.capstone.dto.RequestPhotoDTO;
import com.example.capstone.dto.ResponseAlbumDTO;
import com.example.capstone.dto.ResponsePhotoDTO;
import com.example.capstone.service.AlbumService;
import com.example.capstone.service.FileService;

import java.io.IOException;
import java.util.List;
import java.util.Optional;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.data.crossstore.ChangeSetPersister.NotFoundException;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/albums")
public class AlbumController {

  private final FileService fileService;
  private final AlbumService albumService;

  public AlbumController(FileService fileService, AlbumService albumService) {
    this.fileService = fileService;
    this.albumService = albumService;
  }
  // 파일 업로드
  @PostMapping("/upload")
  public ResponseEntity<Void> uploadPhotos(@ModelAttribute RequestPhotoDTO requestPhotoDTO)
      throws IOException {
    fileService.storeFiles(requestPhotoDTO);
    return new ResponseEntity<>(HttpStatus.OK);
  }

  // 파일 삭제
  @DeleteMapping("/delete")
  public ResponseEntity<Void> deletePhoto(@RequestParam("filePath") String filePath) {
    if (!fileService.doesFileExist(filePath)) {
      return new ResponseEntity<>(HttpStatus.NOT_FOUND);
    }
    fileService.deleteFile(filePath);
    return new ResponseEntity<>(HttpStatus.NO_CONTENT);
  }

  @GetMapping("/{groupId}")
  public ResponseEntity<List<ResponseAlbumDTO>> getAllAlbumData(@PathVariable Long groupId) {
    List<ResponseAlbumDTO> albums = albumService.findAllAlbums(groupId);
    return new ResponseEntity<>(albums, HttpStatus.OK);
  }

  //수정 필요
  @GetMapping("/{groupId}/{albumName}")
  public ResponseEntity<List<ResponsePhotoDTO>> getAlbumData(@PathVariable Long groupId, @PathVariable String albumName) {
    Optional<List<ResponsePhotoDTO>> albumPhotos =
        albumService.findAllAlbumPhotos(new RequestAlbumDTO(groupId, albumName));
    return albumPhotos.map(
            responsePhotoDTOS -> new ResponseEntity<>(responsePhotoDTOS, HttpStatus.OK))
        .orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
  }

  @PostMapping("/{groupId}/{albumName}/download")
  public ResponseEntity<ByteArrayResource> downloadAllAlbumPhotos(@PathVariable Long groupId, @PathVariable String albumName)
      throws IOException {
    Optional<ResponseEntity<ByteArrayResource>> response = albumService.zipAlbum(new RequestAlbumDTO(groupId, albumName));
    return response.orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
  }

  @ExceptionHandler(IOException.class)
  ResponseEntity<String> handleBadSignupRequest(Exception e) {
    return ResponseEntity.badRequest().body(e.getMessage());
  }
}

