param(
    [string]$DatasetRoot = "data/cache/FSD50K",
    [string]$OutputPath = "data/metadata/fsd50k_fire_alarm_candidates.csv"
)

$ErrorActionPreference = "Stop"
$metadataRoot = Join-Path $DatasetRoot "metadata/FSD50K.metadata"
$groundTruthRoot = Join-Path $DatasetRoot "ground_truth/FSD50K.ground_truth"
$allowedReviewLicenses = @(
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by-nc/3.0/",
    "http://creativecommons.org/licenses/by-nc/4.0/"
)

$clipInfo = @{}
foreach ($metadataFile in @("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json")) {
    $jsonPath = Join-Path $metadataRoot $metadataFile
    $json = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json -AsHashtable
    foreach ($entry in $json.GetEnumerator()) {
        $clipInfo[$entry.Key] = $entry.Value
    }
}

$candidates = foreach ($splitName in @("dev", "eval")) {
    $csvPath = Join-Path $groundTruthRoot "$splitName.csv"
    foreach ($row in (Import-Csv -LiteralPath $csvPath)) {
        if ($row.labels -notmatch "(^|,)Alarm(,|$)") { continue }
        $info = $clipInfo[$row.fname]
        if ($null -eq $info) { continue }

        $tags = @($info.tags) -join ";"
        $searchText = "$($info.title) $tags"
        if ($searchText -notmatch "(?i)fire[- _]?alarm|smoke[- _]?alarm") { continue }
        if ($info.license -notin $allowedReviewLicenses) { continue }

        [pscustomobject]@{
            asset_id = $row.fname
            filename = "$($row.fname).wav"
            source_split = $splitName
            labels = $row.labels
            license_url = $info.license
            uploader = $info.uploader
            title = $info.title
            tags = $tags
            candidate_reason = "Alarm label plus fire/smoke alarm wording in title or tags"
            review_status = "needs_audio_content_review"
        }
    }
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$candidates | Sort-Object source_split, asset_id | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding utf8
Write-Output "Wrote $($candidates.Count) candidate rows to $OutputPath"
