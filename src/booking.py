import pandas as pd

# Path to your Excel file
EXCEL_PATH = "data/database/museum_events.xlsx"

def handle_booking(event_name, num_tickets):
    try:
        df = pd.read_excel(EXCEL_PATH)

        match = df[df["Event Name"].str.lower() == event_name.lower()]

        if match.empty:
            return f"❌ Could not find an event named '{event_name}'. Please check the name."

        idx = match.index[0]
        available = int(df.at[idx, "Tickets Left"])

        if available < num_tickets:
            return f"⚠️ Only {available} tickets available for '{event_name}'. Try booking fewer."

        # Reduce tickets and save
        df.at[idx, "Available Tickets"] = available - num_tickets
        df.to_excel(EXCEL_PATH, index=False)

        return f"✅ Successfully booked {num_tickets} tickets for '{event_name}'! Enjoy the event. 🎉"

    except Exception as e:
        return f"❌ Booking failed due to an error: {e}"
