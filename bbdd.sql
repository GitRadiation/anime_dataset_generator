-- Eliminar tablas existentes en el orden correcto (primero la intermedia)
DROP TABLE IF EXISTS user_score;
DROP TABLE IF EXISTS user_details;
DROP TABLE IF EXISTS anime_dataset;

CREATE TABLE anime_dataset (
    anime_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    score NUMERIC(4,2), -- Para valores como 8.75
    genres TEXT[], -- Lista de géneros como array
    keywords TEXT[], -- Lista larga de palabras clave como array
    type VARCHAR(50),
    episodes REAL, -- Puede ser decimal como 26.0
    aired SMALLINT, -- Año como '1998'
    premiered VARCHAR(20), -- Ej: 'spring'
    status VARCHAR(50),
    producers TEXT[], -- Lista de productores como array
    studios VARCHAR(100),
    source VARCHAR(50),
    rating VARCHAR(100),
    rank INTEGER,
    popularity INTEGER,
    favorites INTEGER,
    scored_by INTEGER,
    members INTEGER,
    duration_class VARCHAR(20), -- Nueva columna basada en ejemplo
    episodes_class VARCHAR(20)  -- Nueva columna basada en ejemplo
);

CREATE TABLE user_details (
    mal_id INTEGER PRIMARY KEY,
    username VARCHAR(255), -- Necesario para la relación con user_score
    gender VARCHAR(10),
    age_group VARCHAR(10), -- Grupo de edad: "young", "adult", "senior"
    days_watched NUMERIC(10,2), -- Tiempo en días que el usuario ha visto anime
    mean_score NUMERIC(4,2), -- Promedio de calificación
    watching INTEGER, -- Número de animes que está viendo
    completed INTEGER, -- Número de animes que ha completado
    on_hold INTEGER, -- Número de animes en espera
    dropped INTEGER, -- Número de animes descartados
    plan_to_watch INTEGER, -- Número de animes que planea ver
    total_entries INTEGER, -- Total de entradas en su lista
    rewatched INTEGER, -- Número de animes que ha vuelto a ver
    episodes_watched INTEGER -- Número total de episodios vistos
);

-- Crear tabla user_score (tabla intermedia para la relación N:N entre user_details y anime_dataset)
CREATE TABLE user_score (
    user_id INTEGER,
    anime_id INTEGER,
    rating VARCHAR(10), -- Calificación categorizada: "low", "medium", "high"
    PRIMARY KEY (user_id, anime_id),  -- Clave primaria compuesta
    FOREIGN KEY (user_id) REFERENCES user_details(mal_id),
    FOREIGN KEY (anime_id) REFERENCES anime_dataset(anime_id)
);

DROP TABLE IF EXISTS rules;

CREATE TABLE rules (
    rule_id UUID PRIMARY KEY,
    conditions JSONB NOT NULL,
    target_value TEXT NOT NULL
);

DROP FUNCTION IF EXISTS get_matching_rules(jsonb);

CREATE OR REPLACE FUNCTION get_matching_rules(input JSONB)
RETURNS TABLE(name VARCHAR) AS $$
BEGIN
    RETURN QUERY
    SELECT ad.name
    FROM rules r
    JOIN anime_dataset ad
    ON ad.anime_id = r.target_value
    WHERE (
        -- Condiciones de user_conditions
        SELECT bool_and(
            CASE
            WHEN cond->>'operator' = '>=' THEN 
                (input->'user_conditions'->>(cond->>'column'))::numeric >= (cond->>'value')::numeric
            WHEN cond->>'operator' = '<' THEN 
                (input->'user_conditions'->>(cond->>'column'))::numeric < (cond->>'value')::numeric
            WHEN cond->>'operator' = '==' THEN 
                CASE 
                WHEN jsonb_typeof(input->'user_conditions'->(cond->>'column')) = 'array' THEN
                    EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text(input->'user_conditions'->(cond->>'column')) AS val
                        WHERE val = cond->>'value'
                    )
                ELSE
                    input->'user_conditions'->>(cond->>'column') = cond->>'value'
                END
            ELSE false
            END
        )
        FROM jsonb_array_elements(r.conditions->'user_conditions') AS cond
    )
    AND (
        -- Condiciones de other_conditions
        SELECT bool_and(
            CASE
            WHEN cond->>'operator' = '>=' THEN 
                (input->'other_conditions'->>(cond->>'column'))::numeric >= (cond->>'value')::numeric
            WHEN cond->>'operator' = '<' THEN 
                (input->'other_conditions'->>(cond->>'column'))::numeric < (cond->>'value')::numeric
            WHEN cond->>'operator' = '==' THEN 
                CASE 
                WHEN jsonb_typeof(input->'other_conditions'->(cond->>'column')) = 'array' THEN
                    EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text(input->'other_conditions'->(cond->>'column')) AS val
                        WHERE val = cond->>'value'
                    )
                ELSE
                    input->'other_conditions'->>(cond->>'column') = cond->>'value'
                END
            ELSE false
            END
        )
        FROM jsonb_array_elements(r.conditions->'other_conditions') AS cond
    );
END;
$$ LANGUAGE plpgsql;
