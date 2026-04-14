<?php
// ============================================================
// DATABASE CONFIGURATION
// ============================================================

$host = "localhost";
$user = "root";
$password = "";
$database = "water_safety_ussd";

$conn = new mysqli($host, $user, $password, $database);

if ($conn->connect_error) {
    die("Database connection failed: " . $conn->connect_error);
}
?>
