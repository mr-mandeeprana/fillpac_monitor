/* =============================================================================
   FillPac Operator Monitor - Database Schema
   Run this in SSMS (SQL Server Management Studio) as a user with CREATE
   permissions. Safe to re-run: guarded with IF NOT EXISTS checks.
   ============================================================================= */

IF DB_ID('FillPacMonitor') IS NULL
BEGIN
    CREATE DATABASE FillPacMonitor;
END
GO

USE FillPacMonitor;
GO

-- -----------------------------------------------------------------------------
-- Stations: one row per physical camera/line being monitored
-- -----------------------------------------------------------------------------
IF OBJECT_ID('dbo.Stations', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Stations (
        StationID       VARCHAR(50)   NOT NULL PRIMARY KEY,
        StationName     NVARCHAR(200) NOT NULL,
        Location        NVARCHAR(200) NULL,
        IsActive        BIT           NOT NULL DEFAULT 1,
        CreatedAt       DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME()
    );
END
GO

-- -----------------------------------------------------------------------------
-- StatusEvents: the core event log. One row is written every time the
-- confirmed (debounced) status changes for a tracked operator, plus periodic
-- heartbeat rows so gaps don't look like an outage.
--   Status values: NO_OPERATOR | OPERATOR_PRESENT | NOT_WORKING | WORKING
-- -----------------------------------------------------------------------------
IF OBJECT_ID('dbo.StatusEvents', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.StatusEvents (
        EventID         BIGINT IDENTITY(1,1) PRIMARY KEY,
        StationID       VARCHAR(50)   NOT NULL,
        TrackID         INT           NULL,               -- YOLO track id, NULL if no operator
        Status          VARCHAR(30)   NOT NULL,
        EventType       VARCHAR(20)   NOT NULL DEFAULT 'CHANGE',  -- CHANGE | HEARTBEAT
        EventTimeUtc    DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
        PrevStatus      VARCHAR(30)   NULL,
        PrevDurationSec FLOAT         NULL,                -- how long PrevStatus lasted
        Metadata        NVARCHAR(MAX) NULL,                -- JSON: movement px, hand count, etc.
        CONSTRAINT FK_StatusEvents_Station FOREIGN KEY (StationID)
            REFERENCES dbo.Stations(StationID)
    );

    CREATE INDEX IX_StatusEvents_Station_Time
        ON dbo.StatusEvents (StationID, EventTimeUtc DESC);

    CREATE INDEX IX_StatusEvents_Status
        ON dbo.StatusEvents (Status);
END
GO

-- -----------------------------------------------------------------------------
-- SystemLogs: application-level logs mirrored to DB (errors/warnings mainly;
-- full detail still lives in rotating file logs on disk).
-- -----------------------------------------------------------------------------
IF OBJECT_ID('dbo.SystemLogs', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.SystemLogs (
        LogID           BIGINT IDENTITY(1,1) PRIMARY KEY,
        StationID       VARCHAR(50)   NULL,
        LogLevel        VARCHAR(20)   NOT NULL,
        Component       VARCHAR(100)  NULL,
        Message         NVARCHAR(MAX) NOT NULL,
        LoggedAtUtc     DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME()
    );

    CREATE INDEX IX_SystemLogs_Time ON dbo.SystemLogs (LoggedAtUtc DESC);
END
GO

-- -----------------------------------------------------------------------------
-- CameraHealth: connection / reconnect telemetry
-- -----------------------------------------------------------------------------
IF OBJECT_ID('dbo.CameraHealth', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.CameraHealth (
        HealthID        BIGINT IDENTITY(1,1) PRIMARY KEY,
        StationID       VARCHAR(50)   NOT NULL,
        IsConnected     BIT           NOT NULL,
        ConsecutiveFailures INT       NOT NULL DEFAULT 0,
        ReconnectCount  INT           NOT NULL DEFAULT 0,
        Note            NVARCHAR(500) NULL,
        CheckedAtUtc    DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_CameraHealth_Station FOREIGN KEY (StationID)
            REFERENCES dbo.Stations(StationID)
    );

    CREATE INDEX IX_CameraHealth_Station_Time
        ON dbo.CameraHealth (StationID, CheckedAtUtc DESC);
END
GO

-- -----------------------------------------------------------------------------
-- View: current status per station (latest event)
-- -----------------------------------------------------------------------------
IF OBJECT_ID('dbo.vw_CurrentStatus', 'V') IS NOT NULL
    DROP VIEW dbo.vw_CurrentStatus;
GO

CREATE VIEW dbo.vw_CurrentStatus AS
WITH Latest AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY StationID ORDER BY EventTimeUtc DESC) AS rn
    FROM dbo.StatusEvents
)
SELECT StationID, TrackID, Status, EventTimeUtc AS SinceUtc
FROM Latest
WHERE rn = 1;
GO

-- -----------------------------------------------------------------------------
-- View: daily working-time summary per station (WORKING vs NOT_WORKING vs
-- NO_OPERATOR duration in seconds), based on CHANGE events only.
-- -----------------------------------------------------------------------------
IF OBJECT_ID('dbo.vw_DailyStatusSummary', 'V') IS NOT NULL
    DROP VIEW dbo.vw_DailyStatusSummary;
GO

CREATE VIEW dbo.vw_DailyStatusSummary AS
WITH Changes AS (
    SELECT
        StationID,
        Status,
        EventTimeUtc,
        LEAD(EventTimeUtc) OVER (PARTITION BY StationID ORDER BY EventTimeUtc) AS NextEventTimeUtc
    FROM dbo.StatusEvents
    WHERE EventType = 'CHANGE'
)
SELECT
    StationID,
    CAST(EventTimeUtc AS DATE) AS EventDate,
    Status,
    SUM(DATEDIFF(SECOND, EventTimeUtc, ISNULL(NextEventTimeUtc, SYSUTCDATETIME()))) AS TotalSeconds
FROM Changes
GROUP BY StationID, CAST(EventTimeUtc AS DATE), Status;
GO

-- -----------------------------------------------------------------------------
-- Seed the station used in config.yaml by default (safe upsert)
-- -----------------------------------------------------------------------------
IF NOT EXISTS (SELECT 1 FROM dbo.Stations WHERE StationID = 'FILLPAC-01')
BEGIN
    INSERT INTO dbo.Stations (StationID, StationName, Location)
    VALUES ('FILLPAC-01', 'FillPac Line 1', 'Unspecified');
END
GO
