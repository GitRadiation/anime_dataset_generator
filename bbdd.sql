-- Eliminar tablas existentes en el orden correcto (primero la intermedia)
DROP TABLE IF EXISTS user_score;
DROP TABLE IF EXISTS user_details;
DROP TABLE IF EXISTS anime_dataset;
DROP TABLE IF EXISTS rules;

CREATE TABLE anime_dataset (
    anime_id INTEGER PRIMARY KEY,
    score NUMERIC(3,2),
    genres VARCHAR(255),
    keywords TEXT,
    type VARCHAR(50),
    episodes INTEGER,
    aired VARCHAR(100),
    premiered VARCHAR(50),
    status VARCHAR(50),
    producers VARCHAR(255),
    studios VARCHAR(255),
    source VARCHAR(50),
    duration INTEGER,
    rating VARCHAR(50),
    rank INTEGER,
    popularity INTEGER,
    favorites INTEGER,
    scored_by INTEGER,
    members INTEGER
);

CREATE TABLE user_details (
    mal_id INTEGER PRIMARY KEY,
    gender VARCHAR(10),
    birthday DATE,
    days_watched NUMERIC(10,2),
    mean_score NUMERIC(4,2),
    watching INTEGER,
    completed INTEGER,
    on_hold INTEGER,
    dropped INTEGER,
    plan_to_watch INTEGER,
    total_entries INTEGER,
    rewatched INTEGER,
    episodes_watched INTEGER
);

-- Crear tabla user_score (tabla intermedia para la relación N:N entre user_details y anime_dataset)
CREATE TABLE user_score (
    user_id INTEGER,
    anime_id INTEGER,
    rating INTEGER,
    PRIMARY KEY (user_id, anime_id),  -- Clave primaria compuesta
    FOREIGN KEY (user_id) REFERENCES user_details(mal_id),
    FOREIGN KEY (anime_id) REFERENCES anime_dataset(anime_id)
);

CREATE TABLE rules (
    rule_id UUID PRIMARY KEY,
    conditions JSONB NOT NULL,
    target_column TEXT NOT NULL,
    target_value TEXT NOT NULL
);