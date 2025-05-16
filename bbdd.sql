-- Eliminar tablas existentes en el orden correcto (primero la intermedia)
DROP TABLE IF EXISTS user_score;
DROP TABLE IF EXISTS user_details;
DROP TABLE IF EXISTS anime_dataset;

CREATE TABLE anime_dataset (
    anime_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    score NUMERIC(4,2), -- Para valores como 8.75
    genres TEXT, -- Lista de géneros como string
    keywords TEXT, -- Lista larga de palabras clave
    type VARCHAR(50),
    episodes REAL, -- Puede ser decimal como 26.0
    aired SMALLINT, -- Año como '1998'
    premiered VARCHAR(20), -- Ej: 'spring'
    status VARCHAR(50),
    producers TEXT, -- Lista de productores
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

CREATE OR REPLACE FUNCTION check_and_purge_rules()
RETURNS TABLE(deleted_rule_id UUID, deleted_confidence FLOAT) AS $$
DECLARE
    rule_record RECORD;
    current_confidence FLOAT;
    user_conds TEXT;
    other_conds TEXT;
BEGIN
    CREATE TEMP TABLE IF NOT EXISTS rules_to_delete (
        rule_id UUID,
        confidence FLOAT
    ) ON COMMIT DROP;

    FOR rule_record IN SELECT rule_id, conditions, target_value FROM rules LOOP
        user_conds := build_user_conditions(rule_record.conditions);
        other_conds := build_other_conditions(rule_record.conditions);

        EXECUTE format('
            WITH combined_matches AS (
                SELECT COUNT(*) AS total
                FROM user_details ud
                CROSS JOIN anime_dataset ad
                WHERE %s AND %s
            ),
            target_matches AS (
                SELECT COUNT(*) AS total
                FROM user_details ud
                CROSS JOIN anime_dataset ad
                WHERE %s AND %s
                AND ad.anime_id = %L
            )
            SELECT 
                CASE 
                    WHEN cm.total = 0 THEN 0
                    ELSE tm.total::FLOAT / NULLIF(cm.total, 0)::FLOAT
                END AS confidence
            FROM combined_matches cm, target_matches tm',
            user_conds, other_conds,
            user_conds, other_conds,
            rule_record.target_value
        ) INTO current_confidence;

        IF current_confidence < 0.9 THEN
            INSERT INTO rules_to_delete VALUES (rule_record.rule_id, current_confidence);
        END IF;
    END LOOP;

    RETURN QUERY
    DELETE FROM rules r
    USING rules_to_delete d
    WHERE r.rule_id = d.rule_id
    RETURNING 
    r.rule_id AS "Rule ID",
    d.confidence AS "Confidence Score";
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION build_user_conditions(conditions JSONB)
RETURNS TEXT AS $$
DECLARE
    condition_text TEXT := '';
    cond JSONB;
    column_name TEXT;
    operator TEXT;
    value TEXT;
BEGIN
    FOR cond IN SELECT * FROM jsonb_array_elements(conditions->'user_conditions') LOOP
        column_name := cond->>'column';
        operator := cond->>'operator';
        value := cond->>'value';

        -- Corregir '==' por '='
        IF operator = '==' THEN
            operator := '=';
        END IF;

        -- (Opcional) Validar operadores permitidos
        IF operator NOT IN ('=', '<', '<=', '>', '>=', '!=') THEN
            RAISE EXCEPTION 'Operador no permitido: %', operator;
        END IF;

        IF condition_text <> '' THEN
            condition_text := condition_text || ' AND ';
        END IF;

        -- Detectar si el valor es numérico
        IF value ~ '^\d+(\.\d+)?$' THEN
            condition_text := condition_text || format('ud.%I %s %s', column_name, operator, value);
        ELSE
            condition_text := condition_text || format('ud.%I %s %L', column_name, operator, value);
        END IF;
    END LOOP;
    RETURN condition_text;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION build_other_conditions(conditions JSONB)
RETURNS TEXT AS $$
DECLARE
    condition_text TEXT := '';
    cond JSONB;
    column_name TEXT;
    operator TEXT;
    value TEXT;
BEGIN
    FOR cond IN SELECT * FROM jsonb_array_elements(conditions->'other_conditions') LOOP
        column_name := cond->>'column';
        operator := cond->>'operator';
        value := cond->>'value';

        -- Corregir '==' por '='
        IF operator = '==' THEN
            operator := '=';
        END IF;

        -- (Opcional) Validar operadores permitidos
        IF operator NOT IN ('=', '<', '<=', '>', '>=', '!=') THEN
            RAISE EXCEPTION 'Operador no permitido: %', operator;
        END IF;

        IF condition_text <> '' THEN
            condition_text := condition_text || ' AND ';
        END IF;

        -- Detectar si el valor es numérico
        IF value ~ '^\d+(\.\d+)?$' THEN
            condition_text := condition_text || format('ad.%I %s %s', column_name, operator, value);
        ELSE
            condition_text := condition_text || format('ad.%I %s %L', column_name, operator, value);
        END IF;
    END LOOP;
    RETURN condition_text;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION trigger_check_fitness()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM check_and_purge_rules();
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Triggers mejorados con control de frecuencia
CREATE OR REPLACE TRIGGER user_details_update_trigger
AFTER INSERT OR UPDATE OR DELETE ON user_details
FOR EACH STATEMENT
EXECUTE FUNCTION trigger_check_fitness();

CREATE OR REPLACE TRIGGER anime_dataset_update_trigger
AFTER INSERT OR UPDATE OR DELETE ON anime_dataset
FOR EACH STATEMENT
EXECUTE FUNCTION trigger_check_fitness();