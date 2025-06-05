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

DROP TABLE IF EXISTS rules CASCADE;
DROP TABLE IF EXISTS rule_conditions CASCADE;

-- Esquema corregido de base de datos
CREATE TABLE rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_value INTEGER NOT NULL
);

CREATE TABLE rule_conditions (
    condition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id UUID NOT NULL,
    table_name VARCHAR(50) NOT NULL CHECK (table_name IN ('user_details', 'anime_dataset')), 
    column_name VARCHAR(100) NOT NULL,
    operator VARCHAR(10) NOT NULL CHECK (operator IN ('>=', '<=', '>', '<', '=', '==', '!=')),
    value_text TEXT,
    value_numeric NUMERIC,
    FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE,
    -- Constraint para asegurar que al menos un valor esté presente
    CHECK (
        (value_text IS NOT NULL)::int + 
        (value_numeric IS NOT NULL)::int
    )
);

-- Índices para optimizar consultas
CREATE INDEX idx_rule_conditions_rule_id ON rule_conditions(rule_id);
CREATE INDEX idx_rule_conditions_table_column ON rule_conditions(table_name, column_name);
CREATE INDEX idx_rules_target_value ON rules(target_value);

DROP FUNCTION IF EXISTS get_rules_series;

CREATE OR REPLACE FUNCTION get_rules_series(input_data JSONB) 
RETURNS TABLE(anime_id INT, nombre TEXT, cantidad INTEGER) AS $$
DECLARE
    user_record RECORD;
    anime_record RECORD;
    rule_record RECORD;
    condition_record RECORD;
    user_conditions_met BOOLEAN;
    anime_conditions_met BOOLEAN;
    condition_met BOOLEAN;
    array_check_result BOOLEAN;
    user_data JSONB;
    anime_list JSONB;
    anime_item JSONB;
    temp_anime_id INT;
    temp_anime_name TEXT;
BEGIN
    -- Extraer datos del usuario del JSON
    user_data := input_data->'user';
    anime_list := input_data->'anime_list';
    
    -- Crear tabla temporal para resultados
    CREATE TEMP TABLE temp_results (
        anime_id INT,
        nombre TEXT
    ) ON COMMIT DROP;
    
    -- Iterar sobre cada regla
    FOR rule_record IN 
        SELECT DISTINCT r.rule_id, r.target_value 
        FROM rules r
        INNER JOIN rule_conditions rc ON r.rule_id = rc.rule_id
    LOOP
        -- Verificar condiciones de usuario para esta regla
        user_conditions_met := TRUE;
        
        FOR condition_record IN 
            SELECT * FROM rule_conditions 
            WHERE rule_id = rule_record.rule_id 
            AND table_name = 'user_details'
        LOOP
            condition_met := FALSE;
            
            -- Evaluar condición según el operador
            CASE condition_record.operator
                WHEN '>=' THEN
                    condition_met := (user_data->>condition_record.column_name)::NUMERIC >= condition_record.value_numeric;
                WHEN '<=' THEN
                    condition_met := (user_data->>condition_record.column_name)::NUMERIC <= condition_record.value_numeric;
                WHEN '>' THEN
                    condition_met := (user_data->>condition_record.column_name)::NUMERIC > condition_record.value_numeric;
                WHEN '<' THEN
                    condition_met := (user_data->>condition_record.column_name)::NUMERIC < condition_record.value_numeric;
                WHEN '==' THEN
                    IF condition_record.value_numeric IS NOT NULL THEN
                        condition_met := (user_data->>condition_record.column_name)::NUMERIC = condition_record.value_numeric;
                    ELSE
                        condition_met := (user_data->>condition_record.column_name) = condition_record.value_text;
                    END IF;
                WHEN '!=' THEN
                    IF condition_record.value_numeric IS NOT NULL THEN
                        condition_met := (user_data->>condition_record.column_name)::NUMERIC != condition_record.value_numeric;
                    ELSE
                        condition_met := (user_data->>condition_record.column_name) != condition_record.value_text;
                    END IF;
            END CASE;
            
            -- Si alguna condición de usuario no se cumple, marcar como false
            IF NOT condition_met THEN
                user_conditions_met := FALSE;
                EXIT;
            END IF;
        END LOOP;
        
        -- Si las condiciones de usuario no se cumplen, pasar a la siguiente regla
        IF NOT user_conditions_met THEN
            CONTINUE;
        END IF;
        
        -- Verificar condiciones de anime para esta regla
        -- Iterar sobre cada anime en la lista del usuario
        FOR anime_item IN SELECT * FROM jsonb_array_elements(anime_list)
        LOOP
            anime_conditions_met := TRUE;
            
            -- Verificar todas las condiciones de anime para este anime específico
            FOR condition_record IN 
                SELECT * FROM rule_conditions 
                WHERE rule_id = rule_record.rule_id 
                AND table_name = 'anime_dataset'
            LOOP
                condition_met := FALSE;
                
                -- Evaluar condición según el operador y columna
                CASE condition_record.operator
                    WHEN '>=' THEN
                        condition_met := (anime_item->>condition_record.column_name)::NUMERIC >= condition_record.value_numeric;
                    WHEN '<=' THEN
                        condition_met := (anime_item->>condition_record.column_name)::NUMERIC <= condition_record.value_numeric;
                    WHEN '>' THEN
                        condition_met := (anime_item->>condition_record.column_name)::NUMERIC > condition_record.value_numeric;
                    WHEN '<' THEN
                        condition_met := (anime_item->>condition_record.column_name)::NUMERIC < condition_record.value_numeric;
                    WHEN '==' THEN
                        IF condition_record.value_numeric IS NOT NULL THEN
                            condition_met := (anime_item->>condition_record.column_name)::NUMERIC = condition_record.value_numeric;
                        ELSE
                            -- Para arrays como genres, keywords, producers
                            IF condition_record.column_name IN ('genres', 'keywords', 'producers') THEN
                                -- Verificar si el valor está en el array JSON
                                SELECT bool_or(elem::text = quote_literal(condition_record.value_text)) INTO array_check_result
                                FROM jsonb_array_elements_text(anime_item->condition_record.column_name) elem;
                                condition_met := COALESCE(array_check_result, FALSE);
                            ELSE
                                condition_met := (anime_item->>condition_record.column_name) = condition_record.value_text;
                            END IF;
                        END IF;
                    WHEN '!=' THEN
                        IF condition_record.value_numeric IS NOT NULL THEN
                            condition_met := (anime_item->>condition_record.column_name)::NUMERIC != condition_record.value_numeric;
                        ELSE
                            -- Para arrays como genres, keywords, producers
                            IF condition_record.column_name IN ('genres', 'keywords', 'producers') THEN
                                -- Verificar si el valor NO está en el array JSON
                                SELECT bool_or(elem::text = quote_literal(condition_record.value_text)) INTO array_check_result
                                FROM jsonb_array_elements_text(anime_item->condition_record.column_name) elem;
                                condition_met := NOT COALESCE(array_check_result, FALSE);
                            ELSE
                                condition_met := (anime_item->>condition_record.column_name) != condition_record.value_text;
                            END IF;
                        END IF;
                END CASE;
                
                -- Si alguna condición de anime no se cumple, marcar como false
                IF NOT condition_met THEN
                    anime_conditions_met := FALSE;
                    EXIT;
                END IF;
            END LOOP;
            
            -- Si todas las condiciones de anime se cumplen para este anime, agregar a resultados
            IF anime_conditions_met THEN
                temp_anime_id := (anime_item->>'anime_id')::INT;
                temp_anime_name := COALESCE(anime_item->>'english_name', anime_item->>'name');
                
                INSERT INTO temp_results (anime_id, nombre) 
                VALUES (temp_anime_id, temp_anime_name);
            END IF;
        END LOOP;
    END LOOP;
    
    -- Retornar resultados agrupados con conteo
    RETURN QUERY
    SELECT 
        tr.anime_id,
        tr.nombre,
        COUNT(*)::INTEGER as cantidad
    FROM temp_results tr
    GROUP BY tr.anime_id, tr.nombre
    ORDER BY cantidad DESC, tr.anime_id;
    
END;
$$ LANGUAGE plpgsql;