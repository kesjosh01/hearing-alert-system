param(
    [string]$DatasetRoot = "data/cache/ESC-50/ESC-50-master",
    [string]$OutputPath = "data/metadata/esc50_manifest.csv"
)

$ErrorActionPreference = "Stop"
$metadataPath = Join-Path $DatasetRoot "meta/esc50.csv"
$audioRoot = Join-Path $DatasetRoot "audio"
$splitByFold = @{
    1 = "train"
    2 = "train"
    3 = "train"
    4 = "validation"
    5 = "test"
}
$confuserCategories = @("clock_alarm", "siren", "door_wood_knock", "door_wood_creaks", "glass_breaking")

$records = foreach ($row in (Import-Csv -LiteralPath $metadataPath)) {
    $category = $row.category
    $license = if ($row.esc10 -eq "True") { "CC-BY-3.0" } else { "CC-BY-NC-3.0" }
    $confuser = if ($category -in $confuserCategories) { $category } else { "" }
    $srcFile = $row.src_file
    $sampleId = "ESC50-$($row.filename -replace '\.wav$','')"
    [pscustomobject]@{
        sample_id = $sampleId
        relative_path = (Join-Path $audioRoot $row.filename).Replace("\", "/")
        source_type = "public_dataset"
        source_dataset = "ESC-50"
        source_asset_id = $row.filename
        source_recording_id = $srcFile
        household_id = ""
        device_id = ""
        event_id = $sampleId
        label = "other"
        confuser_tags = $confuser
        overlap = "false"
        recording_date = ""
        session_id = ""
        split = $splitByFold[[int]$row.fold]
        split_group_id = "ESC50-source-$srcFile"
        distance_m = ""
        direction = ""
        occlusion = ""
        room_type = ""
        background_level_dbfs = ""
        sample_rate_hz = "44100"
        channels = "1"
        duration_s = "5"
        license_id = $license
        attribution = "ESC-50 dataset; preserve per-clip credits from upstream LICENSE"
        redistribution_allowed = "noncommercial-with-attribution-check"
        participant_consent_id = ""
        privacy_review = "not-applicable-public-source"
        notes = "Original category=$category; fold=$($row.fold); esc10=$($row.esc10); src_file=$srcFile"
        target_domain = "general"
        upstream_split = $splitByFold[[int]$row.fold]
        split_adjustment_reason = ""
    }
}

# Source IDs must remain grouped even if upstream folds contain exceptions.
# Keep existing held-out material held out; this is a project split, not an
# unmodified ESC-50 benchmark fold assignment.
foreach ($sourceGroup in ($records | Group-Object source_recording_id)) {
    $roles = @($sourceGroup.Group.upstream_split)
    $groupRole = if ($roles -contains "test") { "test" } elseif ($roles -contains "validation") { "validation" } else { "train" }
    foreach ($record in $sourceGroup.Group) {
        $record.split = $groupRole
        if ($record.upstream_split -ne $groupRole) {
            $record.split_adjustment_reason = "source_group_shared_with_more_restrictive_partition"
        }
    }
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$records | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding utf8
Write-Output "Wrote $($records.Count) rows to $OutputPath"
