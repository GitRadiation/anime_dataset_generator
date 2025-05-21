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

drop FUNCTION IF EXISTS get_rules_series;

CREATE OR REPLACE FUNCTION get_rules_series(input_data JSONB)
RETURNS TABLE(nombre TEXT, cantidad INTEGER)
LANGUAGE plpgsql AS
$$
DECLARE
    regla RECORD;
    condicion JSONB;
    cumple BOOLEAN;
    user_data JSONB;
    anime_data JSONB;
    valor_input TEXT;
    operador TEXT;
    columna TEXT;
    valor_regla TEXT;
    target_id INT;
BEGIN
    user_data := input_data->'user';

    CREATE TEMP TABLE reglas_disparadas (
        target_id INT
    ) ON COMMIT DROP;

    FOR anime_data IN SELECT * FROM jsonb_array_elements(input_data->'anime_list') LOOP
        FOR regla IN SELECT * FROM rules LOOP
            cumple := TRUE;

            -- Verificar condiciones de usuario
            FOR condicion IN SELECT * FROM jsonb_array_elements(regla.conditions->'user_conditions') LOOP
                valor_input := user_data->>(condicion->>'column');
                operador := condicion->>'operator';
                valor_regla := condicion->>'value';

                IF operador = '<' THEN
                    IF NOT (valor_input::numeric < valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                ELSIF operador = '<=' THEN
                    IF NOT (valor_input::numeric <= valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                ELSIF operador = '>' THEN
                    IF NOT (valor_input::numeric > valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                ELSIF operador = '>=' THEN
                    IF NOT (valor_input::numeric >= valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                ELSIF operador = '==' THEN
                    IF NOT (valor_input = valor_regla) THEN cumple := FALSE; EXIT; END IF;
                END IF;
            END LOOP;

            -- Verificar condiciones de anime
            IF cumple THEN
                FOR condicion IN SELECT * FROM jsonb_array_elements(regla.conditions->'other_conditions') LOOP
                    valor_input := anime_data->>(condicion->>'column');
                    operador := condicion->>'operator';
                    valor_regla := condicion->>'value';

                    IF operador = '<' THEN
                        IF NOT (valor_input::numeric < valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                    ELSIF operador = '<=' THEN
                        IF NOT (valor_input::numeric <= valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                    ELSIF operador = '>' THEN
                        IF NOT (valor_input::numeric > valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                    ELSIF operador = '>=' THEN
                        IF NOT (valor_input::numeric >= valor_regla::numeric) THEN cumple := FALSE; EXIT; END IF;
                    ELSIF operador = '==' THEN
                        IF NOT (valor_input = valor_regla) THEN cumple := FALSE; EXIT; END IF;
                    END IF;
                END LOOP;
            END IF;

            IF cumple THEN
                target_id := regla.target_value::int;
                INSERT INTO reglas_disparadas VALUES (target_id);
            END IF;
        END LOOP;
    END LOOP;

    RETURN QUERY
    SELECT a.name::TEXT, COUNT(*)::INTEGER AS cantidad
    FROM reglas_disparadas r
    JOIN anime_dataset a ON a.anime_id = r.target_id
    GROUP BY a.name
    ORDER BY cantidad DESC;


END;
$$;
