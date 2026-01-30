#!/usr/bin/env python3
"""
TODO Tracker for LIQUIDEDGE Trading Bot

This script manages project tasks organized by week with priorities and time estimates.

Usage:
    python scripts/todo_tracker.py list              # Show all todos
    python scripts/todo_tracker.py list --week 3     # Show Week 3 todos
    python scripts/todo_tracker.py complete <id>     # Mark todo as complete
    python scripts/todo_tracker.py add               # Interactive add new todo
    python scripts/todo_tracker.py init              # Initialize with default todos
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, date
from dataclasses import dataclass, asdict

# Try to import rich for beautiful output, fallback to basic formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


# ANSI color codes for terminals
class Colors:
    """Terminal color codes."""
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


@dataclass
class Todo:
    """
    TODO item with metadata.

    Attributes:
        id: Unique identifier
        title: Short title of the task
        description: Detailed description
        week: Target week number
        estimated_time: Time estimate (e.g., "45 min", "2 hours")
        priority: Priority level ("critical", "important", "nice-to-have")
        status: Current status ("pending", "in_progress", "completed")
        created_at: Creation timestamp
        completed_at: Completion timestamp (None if not completed)
    """
    id: int
    title: str
    description: str
    week: int
    estimated_time: str
    priority: str
    status: str = "pending"
    created_at: str = ""
    completed_at: Optional[str] = None

    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class TodoTracker:
    """
    TODO tracking system for project management.

    Manages tasks in a JSON file with support for adding, listing,
    and completing tasks with priorities and time estimates.
    """

    def __init__(self, data_file: str = "data/todos.json"):
        """
        Initialize TODO tracker.

        Args:
            data_file: Path to JSON file for storing todos
        """
        self.data_file = Path(data_file)
        self.todos: List[Todo] = []
        self.load()

    def load(self) -> None:
        """Load todos from JSON file."""
        if not self.data_file.exists():
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            self.save()  # Create empty file
            return

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.todos = [Todo(**item) for item in data.get('todos', [])]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading todos: {e}")
            self.todos = []

    def save(self) -> None:
        """Save todos to JSON file."""
        data = {
            'todos': [asdict(todo) for todo in self.todos],
            'last_updated': datetime.now().isoformat()
        }

        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_todo(
        self,
        title: str,
        description: str,
        week: int,
        estimated_time: str,
        priority: str
    ) -> Todo:
        """
        Add a new todo.

        Args:
            title: Task title
            description: Task description
            week: Target week number
            estimated_time: Time estimate
            priority: Priority level

        Returns:
            Created Todo object
        """
        # Get next ID
        next_id = max([t.id for t in self.todos], default=0) + 1

        todo = Todo(
            id=next_id,
            title=title,
            description=description,
            week=week,
            estimated_time=estimated_time,
            priority=priority
        )

        self.todos.append(todo)
        self.save()
        return todo

    def complete_todo(self, todo_id: int) -> bool:
        """
        Mark a todo as completed.

        Args:
            todo_id: ID of todo to complete

        Returns:
            True if successful, False if not found
        """
        for todo in self.todos:
            if todo.id == todo_id:
                todo.status = "completed"
                todo.completed_at = datetime.now().isoformat()
                self.save()
                return True
        return False

    def get_todos_by_week(self, week: Optional[int] = None) -> List[Todo]:
        """
        Get todos for a specific week or all weeks.

        Args:
            week: Week number (None for all weeks)

        Returns:
            List of todos
        """
        if week is None:
            return self.todos

        return [t for t in self.todos if t.week == week]

    def get_overdue_todos(self, current_week: int) -> List[Todo]:
        """
        Get todos that are overdue.

        Args:
            current_week: Current week number

        Returns:
            List of overdue todos
        """
        return [
            t for t in self.todos
            if t.week < current_week and t.status != "completed"
        ]

    def get_priority_color(self, priority: str) -> str:
        """
        Get color for priority level.

        Args:
            priority: Priority level

        Returns:
            Color code string
        """
        if priority == "critical":
            return Colors.RED
        elif priority == "important":
            return Colors.YELLOW
        else:  # nice-to-have
            return Colors.GREEN

    def display_todos(self, todos: List[Todo], title: str = "TODO List") -> None:
        """
        Display todos in a formatted table.

        Args:
            todos: List of todos to display
            title: Table title
        """
        if not todos:
            print(f"\n{Colors.CYAN}No todos found.{Colors.RESET}\n")
            return

        if RICH_AVAILABLE:
            self._display_rich(todos, title)
        else:
            self._display_basic(todos, title)

    def _display_rich(self, todos: List[Todo], title: str) -> None:
        """Display todos using rich library."""
        table = Table(title=title, box=box.ROUNDED)

        table.add_column("ID", style="cyan", width=4)
        table.add_column("Week", style="blue", width=6)
        table.add_column("Title", style="bold", width=30)
        table.add_column("Priority", width=12)
        table.add_column("Time", style="magenta", width=10)
        table.add_column("Status", width=12)

        for todo in todos:
            # Color-code priority
            if todo.priority == "critical":
                priority_style = "bold red"
            elif todo.priority == "important":
                priority_style = "bold yellow"
            else:
                priority_style = "bold green"

            # Status styling
            if todo.status == "completed":
                status_style = "green"
                status_text = "✓ Done"
            elif todo.status == "in_progress":
                status_style = "yellow"
                status_text = "⟳ In Progress"
            else:
                status_style = "white"
                status_text = "○ Pending"

            table.add_row(
                str(todo.id),
                f"Week {todo.week}",
                todo.title,
                f"[{priority_style}]{todo.priority.upper()}[/{priority_style}]",
                todo.estimated_time,
                f"[{status_style}]{status_text}[/{status_style}]"
            )

        console.print(table)

        # Show descriptions in panels
        console.print("\n[bold cyan]Task Details:[/bold cyan]")
        for todo in todos:
            console.print(
                Panel(
                    todo.description,
                    title=f"#{todo.id} - {todo.title}",
                    border_style="cyan"
                )
            )

    def _display_basic(self, todos: List[Todo], title: str) -> None:
        """Display todos using basic terminal formatting."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{title.center(80)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")

        for todo in todos:
            # Priority color
            priority_color = self.get_priority_color(todo.priority)

            # Status symbol
            if todo.status == "completed":
                status = f"{Colors.GREEN}✓ Done{Colors.RESET}"
            elif todo.status == "in_progress":
                status = f"{Colors.YELLOW}⟳ In Progress{Colors.RESET}"
            else:
                status = "○ Pending"

            print(f"{Colors.BOLD}ID #{todo.id} | Week {todo.week} | {status}{Colors.RESET}")
            print(f"{Colors.BOLD}{todo.title}{Colors.RESET}")
            print(f"  {priority_color}Priority: {todo.priority.upper()}{Colors.RESET}")
            print(f"  Estimated time: {todo.estimated_time}")
            print(f"  {todo.description}")
            print(f"{Colors.CYAN}{'-' * 80}{Colors.RESET}\n")


def initialize_default_todos(tracker: TodoTracker) -> None:
    """
    Initialize tracker with default project todos.

    Args:
        tracker: TodoTracker instance
    """
    default_todos = [
        # Week 3
        {
            "title": "Config Management",
            "description": "Create config/assets.yaml and config/parameters.yaml for centralized configuration management",
            "week": 3,
            "estimated_time": "45 min",
            "priority": "important"
        },
        # Week 4
        {
            "title": "Basic Monitoring Dashboard",
            "description": "Build simple dashboard for backtest results visualization with matplotlib/plotly",
            "week": 4,
            "estimated_time": "2 hours",
            "priority": "critical"
        },
        {
            "title": "Update Documentation",
            "description": "Complete README with architecture, setup guide, and troubleshooting section",
            "week": 4,
            "estimated_time": "45 min",
            "priority": "important"
        },
        # Week 5
        {
            "title": "Error Alerting System",
            "description": "Implement Telegram/email alerts for critical errors and trading failures",
            "week": 5,
            "estimated_time": "1 hour",
            "priority": "critical"
        },
        # Week 6
        {
            "title": "Backup Strategy",
            "description": "Automated backup for trade logs, configuration files, and historical data",
            "week": 6,
            "estimated_time": "30 min",
            "priority": "critical"
        },
        {
            "title": "Final Security Audit",
            "description": "Review all code for security issues, credential handling, and API safety before live trading",
            "week": 6,
            "estimated_time": "1 hour",
            "priority": "critical"
        }
    ]

    for todo_data in default_todos:
        tracker.add_todo(**todo_data)

    print(f"{Colors.GREEN}✓ Initialized with {len(default_todos)} default todos{Colors.RESET}")


def interactive_add(tracker: TodoTracker) -> None:
    """
    Interactively add a new todo.

    Args:
        tracker: TodoTracker instance
    """
    print(f"\n{Colors.BOLD}{Colors.CYAN}Add New TODO{Colors.RESET}\n")

    title = input("Title: ").strip()
    if not title:
        print(f"{Colors.RED}Error: Title is required{Colors.RESET}")
        return

    description = input("Description: ").strip()
    if not description:
        print(f"{Colors.RED}Error: Description is required{Colors.RESET}")
        return

    try:
        week = int(input("Week number: ").strip())
    except ValueError:
        print(f"{Colors.RED}Error: Week must be a number{Colors.RESET}")
        return

    estimated_time = input("Estimated time (e.g., '45 min', '2 hours'): ").strip()

    print("\nPriority:")
    print("  1. Critical (must-have)")
    print("  2. Important (should-have)")
    print("  3. Nice-to-have")

    priority_choice = input("Choose priority (1-3): ").strip()

    priority_map = {
        "1": "critical",
        "2": "important",
        "3": "nice-to-have"
    }

    priority = priority_map.get(priority_choice, "important")

    todo = tracker.add_todo(title, description, week, estimated_time, priority)

    print(f"\n{Colors.GREEN}✓ Added TODO #{todo.id}: {todo.title}{Colors.RESET}")


def main():
    """Main entry point for TODO tracker."""
    parser = argparse.ArgumentParser(
        description="LIQUIDEDGE TODO Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/todo_tracker.py list              # Show all todos
  python scripts/todo_tracker.py list --week 3     # Show Week 3 todos
  python scripts/todo_tracker.py complete 5        # Mark todo #5 as done
  python scripts/todo_tracker.py add               # Add new todo
  python scripts/todo_tracker.py init              # Initialize defaults
        """
    )

    parser.add_argument(
        'command',
        choices=['list', 'complete', 'add', 'init'],
        help='Command to execute'
    )

    parser.add_argument(
        'args',
        nargs='*',
        help='Command arguments (e.g., todo ID)'
    )

    parser.add_argument(
        '--week',
        type=int,
        help='Filter by week number'
    )

    args = parser.parse_args()

    # Initialize tracker
    tracker = TodoTracker()

    # Handle commands
    if args.command == 'init':
        initialize_default_todos(tracker)

    elif args.command == 'list':
        if args.week:
            todos = tracker.get_todos_by_week(args.week)
            title = f"TODO List - Week {args.week}"
        else:
            todos = tracker.get_todos_by_week()
            title = "Complete TODO List"

        # Sort by week, then priority
        priority_order = {"critical": 0, "important": 1, "nice-to-have": 2}
        todos.sort(key=lambda t: (t.week, priority_order.get(t.priority, 3)))

        tracker.display_todos(todos, title)

        # Show summary
        total = len(todos)
        completed = len([t for t in todos if t.status == "completed"])
        pending = total - completed

        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"  Total: {total}")
        print(f"  {Colors.GREEN}Completed: {completed}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Pending: {pending}{Colors.RESET}\n")

    elif args.command == 'complete':
        if not args.args:
            print(f"{Colors.RED}Error: Please provide todo ID{Colors.RESET}")
            return

        try:
            todo_id = int(args.args[0])
            if tracker.complete_todo(todo_id):
                print(f"{Colors.GREEN}✓ Marked TODO #{todo_id} as completed{Colors.RESET}")
            else:
                print(f"{Colors.RED}Error: TODO #{todo_id} not found{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Error: Invalid todo ID{Colors.RESET}")

    elif args.command == 'add':
        interactive_add(tracker)


if __name__ == "__main__":
    main()
