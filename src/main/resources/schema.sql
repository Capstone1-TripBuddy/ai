-- User 테이블 생성
CREATE TABLE user (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    profile_picture VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Travel Group 테이블 생성
CREATE TABLE travel_group (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    group_name VARCHAR(255) NOT NULL,
    creator_id BIGINT NOT NULL,
    invite_code VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (creator_id) REFERENCES user(id) ON DELETE CASCADE
);

-- Group Member 테이블 생성
CREATE TABLE group_member (
    group_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_id, user_id),
    FOREIGN KEY (group_id) REFERENCES travel_group(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
);

-- Photo 테이블 생성
CREATE TABLE photo (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    group_id BIGINT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_size INT NOT NULL,
    image_format VARCHAR(50) NOT NULL,
    meta_data TEXT,
    photo_type VARCHAR(50),
    has_face BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
    FOREIGN KEY (group_id) REFERENCES travel_group(id) ON DELETE CASCADE
);

-- Face 테이블 생성
CREATE TABLE face (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    photo_id BIGINT NOT NULL,
    bounding_box TEXT NOT NULL,
    person_id BIGINT NOT NULL,
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);

-- Users Face 테이블 생성
CREATE TABLE user_face (
    user_id BIGINT NOT NULL,
    face_id BIGINT NOT NULL,
    identified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, face_id),
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
    FOREIGN KEY (face_id) REFERENCES face(id) ON DELETE CASCADE
);

-- Album 테이블 생성
CREATE TABLE album (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    group_id BIGINT NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
    FOREIGN KEY (group_id) REFERENCES travel_group(id) ON DELETE CASCADE
);

-- Album Photo 테이블 생성
CREATE TABLE album_photo (
    album_id BIGINT NOT NULL,
    photo_id BIGINT NOT NULL,
    PRIMARY KEY (album_id, photo_id),
    FOREIGN KEY (album_id) REFERENCES album(id) ON DELETE CASCADE,
    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE
);