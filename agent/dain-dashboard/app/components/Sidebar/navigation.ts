import {
  Cpu,
  Activity,
  Settings,
  House,
  MessageSquarePlus,
  Sparkles,
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
    label: "Agent",
    href: "/agent",
    icon: Sparkles,
    description: "Ask the pool, with the cluster as its tools",
  },
  {
    label: "Create Job",
    href: "/create-job",
    icon: MessageSquarePlus,
    description: "Send a new prompt to the cluster",
  },
  {
    label: "Jobs",
    href: "/jobs",
    icon: Activity,
    description: "Monitor status of running jobs",
  },
  
];


// Placeholder, settings page can be updated later with authentication or whatever
// export const secondaryNavigation: NavItem[] = [
//   {
//     label: "Settings",
//     href: "/settings",
//     icon: Settings,
//     description: "Configure D.A.I.N",
//   },
// ];