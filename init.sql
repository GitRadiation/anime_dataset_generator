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
    operator VARCHAR(10) NOT NULL CHECK (operator IN ('>=', '<=', '>', '<', '=', '==', '!=', 'IN', 'NOT IN', 'LIKE', 'ILIKE')),
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

drop FUNCTION IF EXISTS get_rules_series;
CREATE OR REPLACE FUNCTION get_rules_series(input_data JSONB)
RETURNS TABLE(anime_id INT, nombre TEXT, cantidad INTEGER) AS $$
DECLARE
    rule_record RECORD;
    user_data JSONB;
    anime_list JSONB;
    condition_record RECORD;
    rule_matches BOOLEAN;
    user_condition_met BOOLEAN;
    anime_condition_met BOOLEAN;
    anime_item JSONB;
    actual_value TEXT;
    expected_value TEXT;
    elem TEXT;
BEGIN
    user_data := input_data->'user';
    anime_list := input_data->'anime_list';
    
    -- Iterar sobre cada regla única
    FOR rule_record IN 
        SELECT DISTINCT r.rule_id
        FROM rules r
        JOIN rule_conditions rc ON r.rule_id = rc.rule_id
    LOOP
        rule_matches := TRUE;
        
        -- Evaluar condiciones de usuario
        FOR condition_record IN
            SELECT column_name, operator, value_text, value_numeric
            FROM rule_conditions 
            WHERE rule_id = rule_record.rule_id 
            AND table_name = 'user_details'
        LOOP
            user_condition_met := FALSE;
            
            actual_value := user_data->>condition_record.column_name;
            
            expected_value := COALESCE(
                condition_record.value_text,
                condition_record.value_numeric::TEXT
            );
            
            CASE condition_record.operator
                WHEN '>=' THEN
                    user_condition_met := (actual_value::NUMERIC >= condition_record.value_numeric);
                WHEN '<=' THEN
                    user_condition_met := (actual_value::NUMERIC <= condition_record.value_numeric);
                WHEN '>' THEN
                    user_condition_met := (actual_value::NUMERIC > condition_record.value_numeric);
                WHEN '<' THEN
                    user_condition_met := (actual_value::NUMERIC < condition_record.value_numeric);
                WHEN '=' THEN
                    IF condition_record.value_numeric IS NOT NULL THEN
                        user_condition_met := (actual_value::NUMERIC = condition_record.value_numeric);
                    ELSE
                        user_condition_met := (actual_value = condition_record.value_text);
                    END IF;
                WHEN '==' THEN
                    IF condition_record.value_numeric IS NOT NULL THEN
                        user_condition_met := (actual_value::NUMERIC = condition_record.value_numeric);
                    ELSE
                        user_condition_met := (actual_value = condition_record.value_text);
                    END IF;
                WHEN '!=' THEN
                    IF condition_record.value_numeric IS NOT NULL THEN
                        user_condition_met := (actual_value::NUMERIC != condition_record.value_numeric);
                    ELSE
                        user_condition_met := (actual_value != condition_record.value_text);
                    END IF;
                ELSE
                    user_condition_met := FALSE;
            END CASE;
            
            IF NOT user_condition_met THEN
                rule_matches := FALSE;
                EXIT;
            END IF;
        END LOOP;
        
        IF NOT rule_matches THEN
            CONTINUE;
        END IF;
        
        -- Evaluar condiciones de anime para esta regla
        FOR anime_item IN SELECT * FROM jsonb_array_elements(anime_list)
        LOOP
            anime_condition_met := TRUE;
            
            FOR condition_record IN
                SELECT column_name, operator, value_text, value_numeric
                FROM rule_conditions 
                WHERE rule_id = rule_record.rule_id 
                AND table_name = 'anime_dataset'
            LOOP
                actual_value := anime_item->>condition_record.column_name;
                
                -- Comprobación especial para arrays
                IF condition_record.column_name IN ('producers', 'genres', 'keywords') THEN
                    anime_condition_met := FALSE;
                    
                    IF condition_record.operator IN ('=', '==') THEN
                        FOR elem IN SELECT jsonb_array_elements_text(anime_item->condition_record.column_name)
                        LOOP
                            IF elem = condition_record.value_text THEN
                                anime_condition_met := TRUE;
                                EXIT;
                            END IF;
                        END LOOP;
                    ELSIF condition_record.operator = '!=' THEN
                        anime_condition_met := TRUE;
                        FOR elem IN SELECT jsonb_array_elements_text(anime_item->condition_record.column_name)
                        LOOP
                            IF elem = condition_record.value_text THEN
                                anime_condition_met := FALSE;
                                EXIT;
                            END IF;
                        END LOOP;
                    ELSE
                        -- Otros operadores no aplican a arrays
                        anime_condition_met := FALSE;
                    END IF;
                ELSE
                    -- Condiciones normales
                    CASE condition_record.operator
                        WHEN '>=' THEN
                            anime_condition_met := (actual_value::NUMERIC >= condition_record.value_numeric);
                        WHEN '<=' THEN
                            anime_condition_met := (actual_value::NUMERIC <= condition_record.value_numeric);
                        WHEN '>' THEN
                            anime_condition_met := (actual_value::NUMERIC > condition_record.value_numeric);
                        WHEN '<' THEN
                            anime_condition_met := (actual_value::NUMERIC < condition_record.value_numeric);
                        WHEN '=' THEN
                            IF condition_record.value_numeric IS NOT NULL THEN
                                anime_condition_met := (actual_value::NUMERIC = condition_record.value_numeric);
                            ELSE
                                anime_condition_met := (actual_value = condition_record.value_text);
                            END IF;
                        WHEN '==' THEN
                            IF condition_record.value_numeric IS NOT NULL THEN
                                anime_condition_met := (actual_value::NUMERIC = condition_record.value_numeric);
                            ELSE
                                anime_condition_met := (actual_value = condition_record.value_text);
                            END IF;
                        WHEN '!=' THEN
                            IF condition_record.value_numeric IS NOT NULL THEN
                                anime_condition_met := (actual_value::NUMERIC != condition_record.value_numeric);
                            ELSE
                                anime_condition_met := (actual_value != condition_record.value_text);
                            END IF;
                        ELSE
                            anime_condition_met := FALSE;
                    END CASE;
                END IF;
                
                IF NOT anime_condition_met THEN
                    EXIT;
                END IF;
            END LOOP;
            
            IF anime_condition_met THEN
                anime_id := (anime_item->>'anime_id')::INTEGER;
                nombre := anime_item->>'name';
                cantidad := 1;
                RETURN NEXT;
            END IF;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
