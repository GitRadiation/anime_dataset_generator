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
    target_value INTEGER NOT NULL
);

drop FUNCTION IF EXISTS get_rules_series;
CREATE OR REPLACE FUNCTION get_rules_series(input_data JSONB)
RETURNS TABLE(anime_id INT, nombre TEXT, cantidad INTEGER)
LANGUAGE plpgsql AS
$$
DECLARE
    user_data JSONB;
    anime_list JSONB[];
BEGIN
    -- Extraer datos del usuario una sola vez
    user_data := input_data->'user';

    -- Convertir array JSON a array nativo
    SELECT array_agg(value)
    INTO anime_list
    FROM jsonb_array_elements(input_data->'anime_list');

    RETURN QUERY
    WITH user_conditions_eval AS (
        -- Evaluar condiciones de usuario
        SELECT 
            r.rule_id,
            r.target_value::int AS target_id,
            CASE 
                WHEN jsonb_array_length(r.conditions->'user_conditions') = 0 THEN TRUE
                ELSE (
                    SELECT bool_and(
                        CASE uc.value->>'operator'
                            WHEN '<'  THEN (user_data->>(uc.value->>'column'))::numeric < (uc.value->>'value')::numeric
                            WHEN '<=' THEN (user_data->>(uc.value->>'column'))::numeric <= (uc.value->>'value')::numeric
                            WHEN '>'  THEN (user_data->>(uc.value->>'column'))::numeric > (uc.value->>'value')::numeric
                            WHEN '>=' THEN (user_data->>(uc.value->>'column'))::numeric >= (uc.value->>'value')::numeric
                            WHEN '==' THEN (user_data->>(uc.value->>'column')) = (uc.value->>'value')
                            ELSE FALSE
                        END
                    )
                    FROM jsonb_array_elements(r.conditions->'user_conditions') AS uc(value)
                )
            END AS user_conditions_met
        FROM rules r
    ),
    anime_rule_combinations AS (
        -- Generar combinaciones anime-regla
        SELECT 
            uce.rule_id,
            uce.target_id,
            anime_elem.ordinality AS anime_index,
            anime_elem.value AS anime_data
        FROM user_conditions_eval uce
        CROSS JOIN unnest(anime_list) WITH ORDINALITY AS anime_elem(value, ordinality)
        WHERE uce.user_conditions_met = TRUE
    ),
    valid_combinations AS (
        -- Evaluar condiciones de anime
        SELECT 
            arc.target_id,
            arc.anime_data
        FROM anime_rule_combinations arc
        JOIN rules r ON r.rule_id = arc.rule_id
        WHERE 
            CASE 
                WHEN jsonb_array_length(r.conditions->'other_conditions') = 0 THEN TRUE
                ELSE (
                    SELECT bool_and(
                        CASE oc.value->>'operator'
                            WHEN '<'  THEN (arc.anime_data->>(oc.value->>'column'))::numeric < (oc.value->>'value')::numeric
                            WHEN '<=' THEN (arc.anime_data->>(oc.value->>'column'))::numeric <= (oc.value->>'value')::numeric
                            WHEN '>'  THEN (arc.anime_data->>(oc.value->>'column'))::numeric > (oc.value->>'value')::numeric
                            WHEN '>=' THEN (arc.anime_data->>(oc.value->>'column'))::numeric >= (oc.value->>'value')::numeric
                            WHEN '==' THEN
                                CASE
                                    -- Si el campo en la tabla es un array (producers, studios, genres, etc.)
                                    WHEN oc.value->>'column' IN ('producers', 'studios', 'genres', 'keywords') THEN
                                        (arc.anime_data->(oc.value->>'column')) @> to_jsonb(oc.value->>'value'::text)
                                    ELSE
                                        (arc.anime_data->>(oc.value->>'column')) = (oc.value->>'value')
                                END
                            ELSE FALSE
                        END
                    )
                    FROM jsonb_array_elements(r.conditions->'other_conditions') AS oc(value)
                )
            END
    )
    SELECT 
        a.anime_id,
        a.name::TEXT AS nombre,
        COUNT(*)::INTEGER AS cantidad
    FROM valid_combinations vc
    JOIN anime_dataset a ON a.anime_id = vc.target_id
    GROUP BY a.anime_id, a.name
    ORDER BY cantidad DESC;
END;
$$;
