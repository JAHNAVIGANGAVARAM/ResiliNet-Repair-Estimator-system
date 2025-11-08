# Sample Inputs and Expected Outputs

## How to Test the Application

1. Open your browser and go to: **http://localhost:5000**
2. Fill in the form with the examples below
3. Click "Predict" to see results

---

## Sample Input/Output Examples

### Example 1: Fiber Cable Cut
**Input:**
- **Cause**: `fiber cut`
- **Country**: `USA`
- **Region**: `California`
- **Severity**: `high`

**Expected Output:**
- **Predicted Repair Duration**: `3.5 days` (varies based on historical data)
- **Recommended Solution**: `Repair the broken fiber cable in California and reroute traffic via backup path.`

---

### Example 2: DDoS Attack
**Input:**
- **Cause**: `ddos attack`
- **Country**: `India`
- **Region**: `Mumbai`
- **Severity**: `medium`

**Expected Output:**
- **Predicted Repair Duration**: `1.0 days` (typically resolved quickly)
- **Recommended Solution**: `Block attack traffic at the edge and enable DDoS mitigation services.`

---

### Example 3: Power Outage
**Input:**
- **Cause**: `power outage`
- **Country**: `Pakistan`
- **Region**: `Karachi`
- **Severity**: `high`

**Expected Output:**
- **Predicted Repair Duration**: `2.5 days`
- **Recommended Solution**: `Restore power sources (generators/UPS) and restart core network nodes.`

---

### Example 4: Subsea Cable Damage
**Input:**
- **Cause**: `subsea cable`
- **Country**: *(leave empty)*
- **Region**: *(leave empty)*
- **Severity**: `high`

**Expected Output:**
- **Predicted Repair Duration**: `14.0 days` (subsea repairs take longer)
- **Recommended Solution**: `Coordinate with cable operator to repair subsea cable and route via backups.`

---

### Example 5: Government Ordered Shutdown
**Input:**
- **Cause**: `government order`
- **Country**: `Myanmar`
- **Region**: *(leave empty)*
- **Severity**: `high`

**Expected Output:**
- **Predicted Repair Duration**: `7.0 days`
- **Recommended Solution**: `Coordinate with legal teams and restore services when authorized by court.`

---

### Example 6: Network Maintenance
**Input:**
- **Cause**: `maintenance`
- **Country**: *(leave empty)*
- **Region**: *(leave empty)*
- **Severity**: `low`

**Expected Output:**
- **Predicted Repair Duration**: `0.5 days`
- **Recommended Solution**: `Complete maintenance steps and run verification tests to bring services online.`

---

### Example 7: Router Failure
**Input:**
- **Cause**: `router failure`
- **Country**: `UK`
- **Region**: `London`
- **Severity**: `medium`

**Expected Output:**
- **Predicted Repair Duration**: `1.2 days`
- **Recommended Solution**: `Reboot or replace the faulty router and verify routing tables.`

---

### Example 8: Cyber Attack
**Input:**
- **Cause**: `cyber attack`
- **Country**: `Germany`
- **Region**: *(leave empty)*
- **Severity**: `high`

**Expected Output:**
- **Predicted Repair Duration**: `2.0 days`
- **Recommended Solution**: `Isolate affected servers, block malicious IPs, and restore from clean backups.`

---

### Example 9: Social Media Block (WhatsApp)
**Input:**
- **Cause**: `whatsapp block`
- **Country**: `Bangladesh`
- **Region**: *(leave empty)*
- **Severity**: `medium`

**Expected Output:**
- **Predicted Repair Duration**: `3.0 days`
- **Recommended Solution**: `Coordinate with WhatsApp to allow essential messaging and prevent misuse.`

---

### Example 10: ISP Peering Issue
**Input:**
- **Cause**: `isp peering`
- **Country**: `Canada`
- **Region**: `Toronto`
- **Severity**: `low`

**Expected Output:**
- **Predicted Repair Duration**: `0.8 days`
- **Recommended Solution**: `Contact ISPs to fix BGP/peering and restore routing between providers.`

---

## Testing Tips

1. **Fuzzy Matching Works**: Try typos like "fibre cut" or "ddoss attack" - the system will auto-correct
2. **Optional Fields**: You can leave country, region, and severity empty - predictions still work
3. **Case Insensitive**: Works with any case: "FIBER CUT", "fiber cut", "Fiber Cut"
4. **Common Causes to Try**:
   - `fiber cut`
   - `power outage`
   - `ddos attack`
   - `cyber attack`
   - `maintenance`
   - `router failure`
   - `subsea cable`
   - `government shutdown`
   - `protest`
   - `technical issue`

---

## What Happens Behind the Scenes

1. **Fuzzy Matching**: Your input is matched against historical causes
2. **Data Filtering**: System finds similar past incidents based on:
   - Cause (required)
   - Country (if provided)
   - Region (if provided)
3. **Statistical Prediction**:
   - **Duration**: Median of matched historical incidents
   - **Solution**: Most common recommendation for that cause
4. **Fallback**: If no exact matches found, uses global statistics

---

## Expected Response Format

```json
{
  "repair_duration": "3.5 days",
  "recommended_solution": "Repair the broken fiber cable in the area and reroute traffic via backup path."
}
```

---

## Try It Now!

Open **http://localhost:5000** in your browser and test with any of the examples above! 🚀
