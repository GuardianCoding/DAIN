import {
  MessageSquare,
  Cpu,
  Activity,
  Settings,
  Terminal,
  House,
  MessageSquarePlus,
} from "lucide-react";

export type NavItem = {
  label: string;
  href: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
  description?: string;
};

export const navigation: NavItem[] = [
  {
    label: "Home",
    href: "/",
    icon: House,
    description: "Return to landing page",
  },
  {
    label: "Dashboard",
    href: "/dashboard",
    icon: Cpu,
    description: "View connected nodes",
  },
  {
    label: "Prompts",
    href: "/prompts",
    icon: Activity,
    description: "Monitor status of running prompts",
  },
  {
    label: "Create Prompt",
    href: "/new-prompt",
    icon: MessageSquarePlus,
    description: "Send a new prompt to the cluster",
  },
];

export const secondaryNavigation: NavItem[] = [
  {
    label: "Settings",
    href: "/settings",
    icon: Settings,
    description: "Configure D.A.I.N",
  },
];